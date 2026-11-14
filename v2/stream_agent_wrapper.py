import asyncio
import websockets
import json

import gymnasium as gym

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

def _resolve_wrapper_attr(env, name):
    """Resolve an attribute through a Gymnasium wrapper chain without legacy forwarding.

    Gymnasium 0.29's Wrapper.__getattr__ is deprecated. Walking the public
    env links with object.__getattribute__ avoids invoking that fallback
    while still allowing attributes on intermediate wrappers or the base env.
    """
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        try:
            return object.__getattribute__(current, name)
        except AttributeError:
            pass
        try:
            current = object.__getattribute__(current, "env")
        except AttributeError:
            current = None
    raise AttributeError(
        f"{type(env).__name__!s} and its wrapped environments have no attribute {name!r}"
    )


class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(
            self.establish_wc_connection()
        )
        self.upload_interval = 300
        self.steam_step_counter = 0
        self.env = env
        self.coord_list = []
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def __getattr__(self, name):
        """Resolve SB3 VecEnv lookups without deprecated wrapper forwarding.

        SB3 2.3.x uses plain getattr() in SubprocVecEnv.get_attr() and
        env_method(). Gymnasium's get_wrapper_attr() can itself reach the
        deprecated Wrapper.__getattr__ path on nested wrappers in the pinned
        Gymnasium 0.29 dependency, so resolve the wrapper chain directly.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )

        wrapped_env = self.__dict__.get("env")
        if wrapped_env is None:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )

        try:
            return _resolve_wrapper_attr(wrapped_env, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from None

    def step(self, action):

        x_pos = self.emulator.memory[X_POS_ADDRESS]
        y_pos = self.emulator.memory[Y_POS_ADDRESS]
        map_n = self.emulator.memory[MAP_N_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(self.env.seen_coords)}"
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps(
                        {
                          "metadata": self.stream_metadata,
                          "coords": self.coord_list
                        }
                    )
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:
            self.websocket = None
