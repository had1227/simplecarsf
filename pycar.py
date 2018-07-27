# Import a library of functions called 'pygame'
import pygame
import math
import random
import numpy as np

#from rrt_star import rrt_star
from car_model import Car2

# Define some colors
BLACK = (0,   0,   0)
WHITE = (255, 255, 255)
GREEN = (0, 255,   0)
RED = (255,   0,   0)
BLUE = (0,   0, 255)

PI = math.pi

def updateSteering(screen, car):
    pygame.draw.arc(screen, GREEN, [20, 20, 250, 200], PI / 4, 3 * PI / 4, 5)
    pygame.draw.arc(screen, RED, [20, 20, 250, 200], 3 * PI / 4, PI, 5)
    pygame.draw.arc(screen, RED, [20, 20, 250, 200], 0, PI / 4, 5)
    pygame.draw.circle(screen, BLACK, [145, 120], 20)
    # rotate tip of needle from 145,10
    # centered at 145,120
    x1 = 145 - 145
    y1 = 10 - 120
    x2 = x1 * math.cos(car.steering_angle) - y1 * math.sin(car.steering_angle)
    y2 = x1 * math.sin(car.steering_angle) + y1 * math.cos(car.steering_angle)
    x = x2 + 145
    y = y2 + 120
    pygame.draw.line(screen, BLACK, [x, y], [145, 120], 5)

class Point():
    # constructed using a normal tupple
    def __init__(self, point_t = (0,0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])
    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))
    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))
    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))
    def __truediv__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))
    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))
    # get back values in original tuple format
    def get(self):
        return (self.x, self.y)

def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length

    for index in range(0, int(length/dash_length), 2):
        start = origin + (slope *    index    * dash_length)
        end   = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)

def drawRoad(screen):
    # pygame.draw.lines(screen, BLACK, False, [(100,100),(240,100)], 60)

    pygame.draw.lines(screen, BLACK, False, [(50, 300), (1350, 300)], 1)
    draw_dashed_line(screen, BLACK, (50, 350), (1350, 350), 1)
    draw_dashed_line(screen, BLACK, (50, 400), (1350, 400), 1)
    draw_dashed_line(screen, BLACK, (50, 450), (1350, 450), 1)
    pygame.draw.lines(screen, BLACK, False, [(50, 500), (1350, 500)], 1)

    # pygame.draw.arc(screen,BLACK,[210,90,300,300],-PI/2,0,60)
    # pygame.draw.arc(screen,BLACK,[470,100,300,300],0,PI,60)
    # pygame.draw.arc(screen,BLACK,[710,100,300,300],PI,3*PI/2,60)


def updateSpeedometer(screen, car):
    # Select the font to use, size, bold, italics
    font = pygame.font.SysFont('Calibri', 25, True, False)

    # Render the text. "True" means anti-aliased text.
    # Black is the color. This creates an image of the
    # letters, but does not put it on the screen

    if car.gear == "D":
        gear_text = font.render("Gear: Drive", True, BLACK)
    elif car.gear == "STOP":
        gear_text = font.render("Gear: Stopped", True, BLACK)
    elif car.gear == "R":
        gear_text = font.render("Gear: Reverse", True, BLACK)
    else:
        gear_text = font.render("Gear: unknown", True, BLACK)

    # Put the image of the gear_text on the screen
    screen.blit(gear_text, [300, 40])

    speed_text = font.render("Speed: " + str(int(car.speed)/10), True, BLACK)
    screen.blit(speed_text, [300, 60])


def gameLoop(action, car, screen):
    if action == 1 or action == 'a' or action == 'left':
        print('left')
        car.turn(-1)
    elif action == 2 or action == 'd' or action == 'right':
        print('right')
        car.turn(1)


def learningGameLoop():
    print('more code here')

'''
def draw_rrt_path(screen, path):
    for nd in path:
        if(nd.parent != None):
            pygame.draw.line(screen,RED,nd.point,nd.parent.point,1)
'''

class laneFollowingCar1(Car2):
    def __init__(self):
        super().__init__(RED, 60, 385, screen)
        self.car = super().car
        self.car.constant_speed = True
        self.car.speed = 100

class env():
    def __init__(self, visualize=False):

        self.visualize = visualize
        self.size = (1400, 600)

        self.obs_num = 0

        if visualize:
            # Initialize the game engine
            pygame.init()
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("car sim")
            background = pygame.Surface(self.screen.get_size())
            background.fill((0, 0, 0))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        start_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        goal_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        self.start = (random.uniform(50,100), start_line)
        self.goal = (random.uniform(1100,1300), goal_line)
        self.reward = 0
        self.done = False

        self.init_speed = 30
        self.car = Car2(RED, self.start[0], self.start[1], self.screen, speed=self.init_speed)
        self.car.initial_state = (self.start[0], self.start[1], 0, 0, self.init_speed, 0, self.init_speed)
        self.car.speed = self.init_speed
        self.car.vel[0] = self.init_speed
        self.car.reset()

        if self.obs_num!=0:
        
            self.obstacles = [Car2(GREEN, random.uniform(200*(i+1),200*(i+2)), random.uniform(300,500),self.screen) for i in range(4)]

            for obs in self.obstacles:
                obs.gear = 'D'
                obs.speed = random.uniform(10,20)
                obs.constant_speed = True

            self.obs_state1 = [(obs.pose[0], obs.pose[1]) for obs in self.obstacles]
            self.obs_state2 = [obs.speed for obs in self.obstacles]
            self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)
        
        self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.1*self.car.speed)
        #self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.01*(self.goal[0]-self.car.pose[0]), 0.01*(self.goal[1]-self.car.pose[1]))
        #self.state += self.obs_state if self.obs_num!=0 else ()

        self.bound = 20
        self.observation_space = len(self.state)
        self.action_space = 25

        self.time_limit = 100
        self.t = 0

        #self.path_planner = rrt_star(self.screen, self.size, self.bound, self.visualize)

        if self.visualize:
            self.screen.fill(WHITE)

    def reset(self):

        start_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        goal_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        self.start = (random.uniform(50,100), start_line)
        self.goal = (random.uniform(1100,1300), goal_line)

        self.car.initial_state = (self.start[0], self.start[1], 0, 0, self.init_speed, 0, self.init_speed)
        self.car.speed = self.init_speed
        self.car.vel[0] = self.init_speed
        self.car.reset()

        for i in range(4):
            speed = 10*random.random()
            start_line = 325+50*random.randrange(4)+random.uniform(-5,5)
            if self.obs_num!=0:
                self.obstacles[i].initial_state = (random.uniform(200*(i+1)+100,200*(i+1)+200), start_line,0,0,speed,0,speed)
                self.obstacles[i].gear="D"
                self.obstacles[i].constant_speed = True
                self.obstacles[i].reset()
            
        if self.obs_num!=0:
            self.obs_state1 = [(obs.pose[0], obs.pose[1]) for obs in self.obstacles]
            self.obs_state2 = [obs.speed for obs in self.obstacles]
            self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)

        self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.1*self.car.speed)
        #self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.01*(self.goal[0]-self.car.pose[0]), 0.01*(self.goal[1]-self.car.pose[1]))
        #self.state += self.obs_state if self.obs_num!=0 else ()

        self.reward = 0
        self.done = False
        self.t = 0

        if self.visualize:
            self.screen.fill(WHITE)

        return self.state

    def step(self, action):
        """
        steer = min(1, max(action[0], -1))
        accel = min(1, max(action[1], -1))
        """
        steer = 0.5*((action//5)-2)
        accel = 0.5*((action%5)-2)
        self.car.turn(steer)
        self.car.accelerate(accel)

        if self.visualize:
            self.screen.fill(WHITE)
            drawRoad(self.screen)
            #self.road.plotRoad(self.screen)

        rate = 10
        self.car.update(1 / rate)

        if self.obs_num!=0:
            for obs in self.obstacles:
                obs.update(1 / rate)

        if self.visualize:
            updateSteering(self.screen, self.car)
            updateSpeedometer(self.screen, self.car)
            #draw_rrt_path(self.screen, self.path_planner.path)

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self.clock.tick(rate)
        """
        self.obs_state1 = [(obs.pose[0], obs.pose[1]) for obs in self.obstacles]
        self.obs_state2 = [obs.speed for obs in self.obstacles]
        self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)
        """

        self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.1*self.car.speed)
        #self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.01*(self.goal[0]-self.car.pose[0]), 0.01*(self.goal[1]-self.car.pose[1]))

        #self.state = self.state + self.obs_state

        self.reward, self.done, info = self.reward_check()

        self.t += 1

        return self.state, self.reward, self.done, (info=='reached')

    def collision_check(self):
        if self.obs_num ==0:
            return False

        if self.visualize:
            collision = [self.car.rect.colliderect(obs.rect) for obs in self.obstacles]
        else:
            collision = [self.dist(self.car.pose,obs.pose)<self.bound for obs in self.obstacles]

        return any(collision)

    def reward_check(self):

        reward = self.get_reward()

        terminate = False
        info = ''

        if self.done:
            terminate = True
            info = 'done'
        elif self.collision_check():
            terminate = True
            info = 'collision'
        elif(self.car.pose[0]<0):
            terminate = False
            info = 'out of range'
        elif(self.car.pose[1]<300 or self.car.pose[1]>500):
            terminate = False
            info = 'out of range'
        elif(self.t >= self.time_limit):
            terminate = True
            info = 'time out'
        else:
            goal_dist = self.dist(self.car.pose, self.goal)
            if goal_dist<self.bound:
                terminate = True
                info = 'reached'
            else:
                terminate = False
                info = 'on going'

        return reward, terminate, info

    
    def get_reward(self):
        #goal_dist = self.dist(self.car.pose, self.goal)
        goal_dist = self.car.initial_state[0] - self.car.pose[0]

        if self.car.pose[0]>1200:
            reached = True
        else:
            reached = False

        out_of_range = False
        if(self.car.pose[0]<0):
            out_of_range = True
        elif(self.car.pose[1]<300 or self.car.pose[1]>500):
            out_of_range = True

        collision = self.collision_check()

        return -0.001 * goal_dist - 5 * out_of_range - 20 * collision + 100 * reached
    

    def dist(self,p1,p2):     #distance between two points
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        #return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
    '''
    def follow_rrt_path(self):
        if (self.path_planner.path == None):
            self.path_planner.build_rrt_tree(self.car.pose, self.goal)
            path = self.path_planner.get_rrt_path()

        accel = 1
        s1 = self.state

        exp = []
        step = 0
        rewards = 0
        sus = 0
        for nd in self.path_planner.path:
            while(True):
                steering = self.toward_point(nd.point)
                accel = -0.01*self.car.speed + random.gauss(random.uniform(0.5,0.8),0.1)
                accel = min(1, max(accel,-0.2))
                s2,r,d,_ = self.step((steering, accel))
                rewards += r
                step += 1
                a=(steering, accel)
                exp.append((s1,a,r,s2,d))
                if (self.dist(nd.point,self.car.pose)<self.bound or d):
                    break
                s1 = s2
            if(self.reward_check()[0]):
                break
        sus = 1 if (self.reward_check()[1]==100) else 0
        return step, rewards, sus
        #return exp
    '''
    def toward_point(self, point):
        x = point[0] - self.car.pose[0]
        y = point[1] - self.car.pose[1]
        th = math.atan2(y,x)       
        
        steering = th - self.car.angle
        steering = min(1, max(steering,-1))
        return steering


if __name__ == "__main__":
    new_env = env(visualize=True)
    for k in range(100):
        new_env.reset()
        speed = 1
        for i in range(100):
            s, r, d, i = new_env.step(14)
            print (s)
            print (r)
    a=input()
    '''
    total_time = 0
    total_r = 0
    total_s = 0
    for i in range(5):
        new_env.reset()
        new_env.path_planner.build_rrt_tree(new_env.car.pose, new_env.goal)
        new_env.path_planner.get_rrt_path()
        t,r,s = new_env.follow_rrt_path()
        total_time += t
        total_r += r
        total_s += s

    print ('Avg Reward: ',total_r/200,'Avg Success: ',total_s/200,'Avg Step: ',total_time/200)
    np.save('rrtstar_result.npy',np.asarray([[total_r/200,total_s/200,total_time/200]]))

    for i in range(20000):
        s, r, d, _ = new_env.step((random.uniform(-1,1),random.uniform(0,0)))

        if(d):
            print(new_env.t)
            print(i)
            new_env.reset()
            print(r)
            break
    '''
