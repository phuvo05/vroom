"""Physical hazards and seeded scenery, visible to both the agent and demo."""

import math
from dataclasses import dataclass

import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing, TRACK_WIDTH


SCENARIOS = ("classic", "forest", "city", "desert")
DIFFICULTIES = ("normal", "hard")
WEATHER = ("dry", "rain")


LOOKAHEAD_TILES = 10


def speed_penalty(speed_kmh, curvature):
    """Per-frame shaping: speed above what the upcoming bend allows is penalised.

    `curvature` is the heading change (rad) over the next LOOKAHEAD_TILES tiles. Measured
    on the first model: ~220 km/h even where it is >1 rad, with +3.5 reward per agent step.
    """
    safe = min(220., max(60., 220. - 110. * curvature))
    return .01 * max(0., speed_kmh - safe)


@dataclass
class Obstacle:
    body: object
    kind: str
    half_width: float
    half_length: float
    hit: bool = False


class ChallengeCarRacing(CarRacing):
    def __init__(self, *, scenario="forest", difficulty="normal", weather="dry",
                 render_mode="rgb_array"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        if difficulty not in DIFFICULTIES or weather not in WEATHER:
            raise ValueError("Invalid difficulty or weather")
        self.scenario, self.difficulty, self.weather = scenario, difficulty, weather
        self.obstacles, self.scenery, self.puddles = [], [], []
        self.damage = self.collisions = 0
        super().__init__(render_mode=render_mode, continuous=False)

    def _destroy(self):
        for obstacle in self.obstacles:
            self.world.DestroyBody(obstacle.body)
        self.obstacles.clear()
        self.scenery.clear()
        self.puddles.clear()
        self.damage = self.collisions = 0
        super()._destroy()

    def _create_track(self):
        if not super()._create_track():
            return False
        if self.scenario == "classic" and self.weather == "dry":
            return True  # Preserve original pixels and physics for old weights.
        for tile in self.road:
            tile.road_friction = 0.62 if self.weather == "rain" else 1.0
        if self.weather == "rain":
            for index in range(24, len(self.track) - 12, 39):
                for tile in self.road[index:index + 3]:
                    tile.road_friction = 0.32
                    self.puddles.append(tile.idx)
        if self.scenario != "classic":
            self._place_obstacles()
            self._place_scenery()
        self._track_index = 0
        return True

    def _upcoming_curvature(self):
        count = len(self.track)
        x, y = self.car.hull.position
        window = [(self._track_index + k) % count for k in range(-3, 12)]
        self._track_index = min(window, key=lambda i: (self.track[i][2] - x) ** 2 + (self.track[i][3] - y) ** 2)
        turn = self.track[(self._track_index + LOOKAHEAD_TILES) % count][1] - self.track[self._track_index][1]
        return abs((turn + math.pi) % math.tau - math.pi)

    def _place_obstacles(self):
        spacing = 13 if self.difficulty == "hard" else 23
        # Reserve the start and finish. A 3.2m barrier leaves room on a 13.3m road.
        for number, index in enumerate(range(14, len(self.track) - 14, spacing)):
            _, beta, x, y = self.track[index]
            side = -1 if number % 2 == 0 else 1
            offset = side * self.np_random.uniform(.24, .43) * TRACK_WIDTH
            position = (x + offset * math.cos(beta), y + offset * math.sin(beta))
            kind, width, length = [("barrier", 1.6, .65), ("parked_car", 1., 2.),
                                   ("crate", .95, .95)][number % 3]
            body = self.world.CreateStaticBody(position=position, angle=beta)
            body.CreatePolygonFixture(box=(width, length), friction=.6, restitution=.05)
            # No userData: the base road-contact listener ignores these bodies.
            self.obstacles.append(Obstacle(body, kind, width, length))

    def _place_scenery(self):
        centers = np.array([(t[2], t[3]) for t in self.track])
        for index in range(0, len(self.track), 4):
            _, beta, x, y = self.track[index]
            for side in (-1, 1):
                distance = self.np_random.uniform(11, 22)
                px, py = x + side * distance * math.cos(beta), y + side * distance * math.sin(beta)
                # Exclude every road section, including the other side of hairpins.
                if np.min(np.linalg.norm(centers - (px, py), axis=1)) < TRACK_WIDTH + 4:
                    continue
                self.scenery.append((px, py, beta, self.np_random.uniform(2, 3.8),
                                     int(self.np_random.integers(3))))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if action is not None and self.scenario != "classic":
            # Classic keeps the original reward so old weights stay comparable.
            reward -= speed_penalty(float(np.linalg.norm(self.car.hull.linearVelocity)) * 3.6,
                                    self._upcoming_curvature())
        if action is not None:
            car_bodies = [self.car.hull, *self.car.wheels]
            for obstacle in self.obstacles:
                if obstacle.hit:
                    continue
                touched = any(edge.contact.touching and any(edge.other == b for b in car_bodies)
                              for edge in obstacle.body.contacts)
                if touched:
                    obstacle.hit = True
                    self.collisions += 1
                    self.damage = min(100, self.damage + (50 if self.difficulty == "hard" else 35))
                    reward -= 80.
            if self.damage >= 100:
                terminated = True
                info["end_reason"] = "wrecked"
            elif terminated:
                info["end_reason"] = "finished" if info.get("lap_finished") else "out_of_bounds"
        info.update(scenario=self.scenario, weather=self.weather,
                    collisions=self.collisions, damage=self.damage,
                    speed_kmh=float(np.linalg.norm(self.car.hull.linearVelocity)) * 3.6,
                    progress=self.tile_visited_count / len(self.track),
                    obstacle_count=len(self.obstacles), car_position=tuple(self.car.hull.position))
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _box(x, y, angle, width, length):
        co, si = math.cos(angle), math.sin(angle)
        return [(x + dx * co - dy * si, y + dx * si + dy * co)
                for dx, dy in [(-width, -length), (width, -length), (width, length), (-width, length)]]

    @staticmethod
    def _circle(x, y, radius):
        return [(x + radius * math.cos(i * math.tau / 12),
                 y + radius * math.sin(i * math.tau / 12)) for i in range(12)]

    def _render_road(self, zoom, translation, angle):
        if self.scenario == "classic":
            super()._render_road(zoom, translation, angle)
            for index in self.puddles:
                _, beta, x, y = self.track[index]
                self._draw_colored_polygon(self.surf, self._box(x, y, beta, TRACK_WIDTH * .8, 1.5),
                                           (90, 126, 145), zoom, translation, angle)
            return
        ground, patch = {"forest": ((95, 154, 102), (109, 169, 110)),
                         "city": ((155, 165, 157), (169, 176, 165)),
                         "desert": ((192, 158, 105), (207, 175, 120))}[self.scenario]
        self.surf.fill(ground)

        def draw(poly, color):
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

        car_x, car_y = self.car.hull.position
        for x, y, beta, size, variant in self.scenery:
            if (x - car_x) ** 2 + (y - car_y) ** 2 > 95 ** 2:
                continue
            draw(self._circle(x + 1, y - 1, size * 1.7), patch)
            if self.scenario == "forest":
                draw(self._circle(x + .6, y - .6, size * 1.05), (29, 53, 42))
                draw(self._circle(x, y, size), (26, 69 + variant * 9, 48))
                draw(self._circle(x - size * .25, y + size * .2, size * .65), (55, 119, 70))
            elif self.scenario == "city":
                draw(self._box(x + .6, y - .7, beta, size + .4, size * 1.4), (67, 77, 83))
                draw(self._box(x, y, beta, size, size * 1.4),
                     [(166, 173, 178), (151, 134, 117), (121, 148, 163)][variant])
                draw(self._box(x, y, beta, size * .7, size), (74, 91, 104))
                for shift in (-1, 1):
                    dx, dy = shift * size * .45 * math.cos(beta), shift * size * .45 * math.sin(beta)
                    draw(self._box(x + dx, y + dy, beta, .22, size * .65), (157, 208, 215))
            else:
                draw(self._circle(x + .4, y - .4, size), (153, 119, 83))
                draw(self._circle(x, y, size * .85), (173 + variant * 9, 139, 98))
                draw(self._circle(x - .3, y + .3, size * .45), (212, 180, 131))
        for poly, color in self.road_poly:
            color = tuple(int(c) for c in color)
            if max(color) - min(color) < 10:
                color = (102, 102, 102) if self.weather == "dry" else (86, 96, 104)
            draw(poly, color)
        for index in range(0, len(self.track), 3):
            _, beta, x, y = self.track[index]
            draw(self._box(x, y, beta, .12, .95), (220, 211, 157))
            for side in (-1, 1):
                px, py = x + side * (TRACK_WIDTH - .25) * math.cos(beta), y + side * (TRACK_WIDTH - .25) * math.sin(beta)
                draw(self._box(px, py, beta, .10, 1.6), (213, 221, 220))
        for index in self.puddles:
            _, beta, x, y = self.track[index]
            draw(self._box(x, y, beta, TRACK_WIDTH * .8, 1.5), (90, 126, 145))
        for obstacle in self.obstacles:
            x, y = obstacle.body.position
            if (x - car_x) ** 2 + (y - car_y) ** 2 > 95 ** 2:
                continue
            beta, w, h = obstacle.body.angle, obstacle.half_width, obstacle.half_length
            draw(self._box(x + .3, y - .3, beta, w + .12, h + .12), (28, 35, 41))
            if obstacle.kind == "barrier":
                draw(self._box(x, y, beta, w, h), (243, 134, 48))
                for offset in (-.95, 0, .95):
                    draw(self._box(x + offset * math.cos(beta), y + offset * math.sin(beta), beta, .18, h), (249, 234, 204))
            elif obstacle.kind == "parked_car":
                draw(self._box(x, y, beta, w, h), (76, 145, 194))
                draw(self._box(x, y, beta, w * .75, h * .5), (29, 56, 78))
                draw(self._box(x, y, beta, w * .55, h * .25), (147, 207, 218))
            else:
                draw(self._box(x, y, beta, w, h), (184, 127, 65))
                draw(self._box(x, y, beta, w * .8, h * .8), (220, 163, 88))
                draw(self._box(x, y, beta, .13, h), (124, 82, 48))
            if obstacle.hit:
                draw(self._circle(x, y, .35), (248, 70, 77))
