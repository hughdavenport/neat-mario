import retro
import numpy as np
import math

class SuperMarioBros:

    KEYS = {
        "B": 0,
        "DOWN": 1,
        "LEFT": 2,
        "RIGHT": 3,
        "A": 4
    }

    TILES = {
        # 60 is hidden 1up
        # 5f is hidden coin
        "NOTHING": [0x00, 0x60, 0x5f],

        # 08 is vine up above
        # 16, 17, 18 are green treetop things
        # 22 is weed in water
        # 54 is broken ground (bottom of level 1)
        # 61 is the square tiles next to flagpole
        # 62 is white brick in bowser dungeons
        # 63 is bridge
        # 69 is ground in water
        # 88 is a cloud
        # 89 is bowser bridge
        "GROUND": [0x08, 0x16, 0x17, 0x18, 0x22, 0x54, 0x61, 0x62, 0x63, 0x69, 0x88, 0x89] +
        # 65 is bottom of bullet firer with warning symbol
        # 66 is the bottom of bullet firer that is plain
                [0x65, 0x66] +
        # 14 and 15 are bottom of pipe so treat as ground
                [0x14, 0x15] +
        # 1d and 20 appear to be top and bottom of horizontal pipe so treat as ground
        # 1e and 21 appear to be the top and bottom the join to vertical so treat as ground
                [0x1d, 0x20, 0x1e, 0x21] +
        # c4 is used box
                [0xc4] +
        # 68 is bottom of spring so treat as ground
                [0x68] +
                [],

        # c0 is coin box
        # c1 is powerup box
        "BOX": [0xc0, 0xc1],

        # c2 is coin
        # c3 is coin in water
        "COIN": [0xc2, 0xc3],

        # 51 is normal brick, 23 is slightly moved up brick (ie when hit from below)
        # 52 is dungeon bricks (secret in 1-1, main in 1-2)
        # 55 is hidden mushroom brick
        # 56 is hidden vine
        # 57 is star brick
        # 58 is repeat coin brick
        # 5a is hidden mushroom in dungeon
        # 5b is ??
        # 5c is star brick in dungeon
        # 5d is bouncing brick in 1-2
        # 5e is hidden 1up in dungeon
        "BRICK": [0x23, 0x51, 0x52, 0x55, 0x56, 0x57, 0x58, 0x5a] +
            # remove this for now to find out what it is
            #, 0x5b, 
            [0x5c, 0x5d, 0x5e],

        "VINE": [0x26],

        # 67 is top
        # 68 is bottom of spring so treat as ground
        "SPRING": [0x67],

        # 64 is shooter
        # 65 and 66 are the bottom of the shooter so treat as ground
        "SHOOTER": [0x64],


        # 10 and 11 are top left and right of teleport pipe
        # 12 and 13 are top left and right of normal pipe,
        # 14 and 15 are bottom of pipe so treat as ground
        "PIPETOP": [0x10, 0x11, 0x12, 0x13],

        # 1c and 1f are top and bottom left of sideways pipe
        # 6b and 6c are top and bottom left of sideways pipe in water
        # 1d and 20 appear to be top and bottom of horizontal pipe so treat as ground
        # 1e and 21 appear to be the top and bottom the join to vertical so treat as ground
        "PIPELEFT": [0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x6b, 0x6c],

        "FLAGPOLE": [0x25],
        "FLAGPOLE_TOP": [0x24],
    }

    SPRITES = {
        # 13 is "nothing"
        # 16 is fireworks
        # 31 is star flag on top of castle at end
        "NOTHING": [0x13, 0x16, 0x31],

        # 06 is Goomba
        "GOOMBA": [0x06],

        # 00 is green koopa
        # what is diff between koopas?
        "KOOPA": [0x00],
    }

    THINGS = list(dict.fromkeys(list(TILES.keys()) + list(SPRITES.keys()) + ["MUSHROOM", "FLOWER", "STAR", "1UP"]))


    def __init__(self):
        self.env = retro.make('SuperMarioBros-Nes', 'Level1-1')
        self.reset()

    def reset(self):
        self._fitness_offset = 0
        self.ob = self.env.reset()
        self.info = self.env.data.lookup_all()

    def step(self, keys=[0, 0, 0, 0, 0]):
#        if keys[SuperMarioBros.KEYS["RIGHT"]] == 1:
#            print("RIGHT")
#        if keys[SuperMarioBros.KEYS["A"]] == 1:
#            print("JUMP")
#        if keys[SuperMarioBros.KEYS["B"]] == 1:
#            print("RUN")
        #           b       ?   st  se  up    down, left, right, a
        keys = [keys[0]] + [0., 0., 0., 0.] + keys[1:5]
        self.ob, _, _, self.info = self.env.step(keys)

    def isFinished(self):
        return self.info['state'] == 0x0B or self.info['viewport'] > 1

    def score(self):
        return self.info['score']

    def time(self):
        return self.info['time']

    def render(self):
        self.env.render()

    def close(self):
        self.env.render(close=True)
        self.env.close()

    def screen(self):
        return self.ob

    def state(self):
        return np.concatenate((self.playerData(), self.playerView(5)))

    def isControllable(self):
        if self.info['state'] == 0x01:
            # vine
            # TODO, how does fitness change during this zone
            pass
        elif self.info['state'] == 0x02:
            # side pipe, coming back to surface
            self._fitness_offset = 0
        elif self.info['state'] == 0x03:
            # downpipe, going underground
            self._fitness_offset = self._playerX()

        return self.info['state'] == 0x08

    def fitness(self):
        return self.info['levelHi'] * 4 + self.info['levelLo'] + self._playerX() + self._fitness_offset

    def tiles(self):
        """Returns tiles as a 15x13 array"""

        ret = np.zeros(shape=(15, 13))
        sx = self.info['xscrollHi'] * 256 + self.info['xscrollLo'] + 8

        # In reality, we can't quite see the next tile off screen (though can sometimes see half)
        for tx in range(sx, sx + 240, 16):
            # In reality, the top 2 and bottom 1 tile is not playable
            for ty in range(32, 240, 16):
                x = int((tx - sx)/16)
                y = int((ty - 32)/16)

                tile = self._getTileData(tx, ty)
                value = None
                for thing in SuperMarioBros.TILES.keys():
                    if tile in SuperMarioBros.TILES[thing]:
                        value = SuperMarioBros.THINGS.index(thing) / (len(SuperMarioBros.THINGS) - 1)
                if value is None:
                    print("unknown tile at (%d, %d) = %x" % (tx, ty, tile))
                ret[x][y] = value

        return ret

    def sprites(self):
        """Returns sprites on screen as a 15x13 array"""

        ret = np.full(shape=(15, 13), fill_value=None)
        sx = self.info['xscrollHi'] * 256 + self.info['xscrollLo'] + 8

        for slot in range(0, 5):
            if self.info['enemy%d?' % slot] == 1:
                ex = self.info['enemy%d_level_x' % slot] * 256 + self.info['enemy%d_screen_x' % slot] + 8
                ey = self.info['enemy%d_screen_y' % slot] + 8
                if ex >= sx and ex < sx + 240 and ey >= 32 and ey < 256 - 16:
                    x = int((ex - sx)/16)
                    y = int((ey - 32)/16)

                    sprite = self.info['enemy%d_type' % slot]
                    value = None
                    for thing in SuperMarioBros.SPRITES.keys():
                        if sprite in SuperMarioBros.SPRITES[thing]:
                            value = SuperMarioBros.THINGS.index(thing) / (len(SuperMarioBros.THINGS) - 1)
                    if value is None:
                        print("unknown sprite at (%d, %d) = %x" % (ex, ey, sprite))
                    ret[x][y] = value

        if self.info['powerup?'] == 1:
            ex = self.info['powerup_level_x'] * 256 + self.info['powerup_screen_x']
            ey = self.info['powerup_screen_y'] + 8
            if ex >= sx and ex < sx + 240 and ey >= 32 and ey < 256 - 16:
                x = int((ex - sx)/16)
                y = int((ey - 32)/16)

                powerup = self.info['powerup_type']
                value = None
                if powerup == 0:
                    value = SuperMarioBros.THINGS.index("MUSHROOM") / (len(SuperMarioBros.THINGS) - 1)
                elif powerup == 1:
                    value = SuperMarioBros.THINGS.index("FLOWER") / (len(SuperMarioBros.THINGS) - 1)
                elif powerup == 2:
                    value = SuperMarioBros.THINGS.index("STAR") / (len(SuperMarioBros.THINGS) - 1)
                elif powerup == 3:
                    value = SuperMarioBros.THINGS.index("1UP") / (len(SuperMarioBros.THINGS) - 1)
                else:
                    print("unknown powerup at (%d, %d) = %x" % (ex, ey, powerup))

                # The flagpole weirdly shows as mushroom, and is sometimes offset
                flagpole = False
                for offset in range(8, 24):
                    add = int((ex - sx + offset)/16) * 16
                    if self._getTileData(sx + add, ey - 8) in SuperMarioBros.TILES["FLAGPOLE"]:
                        flagpole = True
                if not flagpole:
                    if ret[x][y] is not None:
                        print("overwriting a sprite with a powerup at (%d, %d) from %f (%s) to %f (%s)" % (ex, ey, ret[x][y], SuperMarioBros.THINGS[int(ret[x][y] * (len(SuperMarioBros.THINGS) - 1))], value, SuperMarioBros.THINGS[int(value * (len(SuperMarioBros.THINGS) - 1))]))
                    ret[x][y] = value


        # TODO fireball, hammer

        return ret

    def playerData(self):
        """Return data about player, ie big/small"""

        # Could have big/small, in starmode? ducking? speed? floating state?

        ret = np.zeros(shape=0)
        # TODO these info (and others in this file) are based off a custom json file. We could do the hex values direct here
        # Are we small (0) or big (1), or fiery (2)?
        ret = np.append(ret, self.info['powerup_state'] / 3)
        # Are we normal (0), jumping (1)  or falling (2) or coming down flagpole (3)
        ret = np.append(ret, self.info['float'] / 4)
        # Speed left (-) or right (+) up to 127
        ret = np.append(ret, (self.info['speed'] + 127) / 256)


        return ret

    def playerView(self, radius=1):
        """Return spiral view out from player to radius"""
        # This formula for size is based on progression of adding 4 edges and 4 corners each time, with edge growing by 2 each time, then simplified, ignoring tile we are standing on
        ret = np.zeros(shape=(4*(radius**2 - radius)))

        if self._playerY() < 24 or self._playerY() > 232:
            # off top or bottom of screen, return nothing as not playable
#            print("TODO check that we can't play here (%d, %d)" % (self._playerX(), self._playerY()))
            return ret

        # Might just be self._playerX() - sx + 8
        # Need to work out what happens after a few screens
        sx = self.info['xscrollHi'] * 256 + self.info['xscrollLo'] + 8
        y = math.floor((self._playerY() - 24)/16)
        x = math.floor((self._playerX() - sx + 8) / 16)
        tiles = self.tiles()
        sprites = self.sprites()
        # TODO enemies, hammers, mushrooms etc (sprites?)
        index = 0
        for i in range(1, radius):
            # Go up one, then right 2*i - 1, then corner
            y -= 1
            if y < 0 or y >= len(tiles[0]):
                ret[index:(index + 2*i)] = 0.
                index += 2*i
                x += 2*i - 1
            else:
                for x in range(x, x + 2*i):
                    ret[index] = 0. if x >= len(tiles) or x < 0 else tiles[x][y]
                    ret[index] = ret[index] if x >= len(sprites) or x < 0 or sprites[x][y] is None else sprites[x][y]
                    index += 1
            # then down 2*i - 1, then corner
            if x >= len(tiles) or x < 0:
                ret[index:(index + 2*i)] = 0.
                index += 2*i
                y += 2*i
            else:
                for y in range(y + 1, y + 2*i + 1):
                    ret[index] = 0. if y >= len(tiles[x]) or y < 0 else tiles[x][y]
                    ret[index] = ret[index] if y >= len(sprites[x]) or y < 0 or sprites[x][y] is None else sprites[x][y]
                    index += 1
            # then left 2*i - 1, then corner
            if y >= len(tiles[0]) or y < 0:
                ret[index:(index + 2*i)] = 0.
                index += 2*i
                x -= 2*i
            else:
                for x in range(x - 1, x - 2*i - 1, -1):
                    ret[index] = 0. if x < 0 or x >= len(tiles) else tiles[x][y]
                    ret[index] = ret[index] if x < 0 or x >= len(sprites) or sprites[x][y] is None else sprites[x][y]
                    index += 1
            # then up 2*i - 1, then corner
            if x < 0 or x >= len(tiles):
                ret[index:(index + 2*i)] = 0.
                index += 2*i
                y -= 2*i
            else:
                for y in range(y - 1, y - 2*i - 1, -1):
                    ret[index] = 0. if y < 0 or y >= len(tiles[x]) else tiles[x][y]
                    ret[index] = ret[index] if y < 0 or y >= len(sprites[x]) or sprites[x][y] is None else sprites[x][y]
                    index += 1
            # (x, y) should be transformed by (-1, -1) from start of loop

        return ret

    def _getTileData(self, x, y):
        # 0 is unknown, 1 is no tile, 2 is tile
        page = math.floor(x/256)%2
        subx = math.floor((x%256)/16)
        suby = math.floor((y - 32)/16)
        addr = 0x500 + page*13*16+suby*16+subx

        #if suby >= 13 or suby < 0:
        if addr < 0x500 or addr > 0x69F:
            # unknown
            print("bad address")
            return None

        return self.env.data.memory.extract(addr, ">u1")

    def _playerX(self):
        return self.info['player_level_x'] * 256 + self.info['player_screen_x']

    def _playerY(self):
        return (self.info['viewport'] - 1) * 256 + self.info['player_screen_y'] + 8
