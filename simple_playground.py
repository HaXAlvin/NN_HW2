import math as m
import random as r
from simple_geometry import Line2D, Point2D
from matplotlib import pyplot as plt
from model import RBF
import numpy as np
np.random.seed(1234)

class Car():
    def __init__(self) -> None:
        self.radius = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def diameter(self):
        return self.radius/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        if self.wheel_min <= angle <= self.wheel_max:
            self.wheel_angle = angle
        elif angle <= self.wheel_min:
            self.wheel_angle = self.wheel_min
        else:
            self.wheel_angle = self.wheel_max

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)

        # seem as a car
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / self.radius))*180/m.pi

        # seem as a circle
        # new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) /
        #              (self.radius)))*180/m.pi

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground():
    def __init__(self, model:RBF=None):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._readPathLines()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]
        self.model = model
        self.car = Car()
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    def predictAction(self, state):
        # state = [front right left]
        if self.model is not None:
            if self.model.indim == 4:
                state = [self.car.xpos,self.car.ypos, state[0], state[2]-state[1]]
            else:
                state = [state[0], state[2]-state[1]]
            return self.model.predict(state)+40
        return r.randint(0, self.n_actions-1)

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            'front').distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            'right').distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            'left').distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # check every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + action*(self.car.wheel_max-self.car.wheel_min) / (self.n_actions-1)
        # print(angle,self.car.wheel_min+action)
        return angle

    def step(self, action=None):
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state


def run_example(p:Playground):
    # use example, select random actions until gameover
    
    path = {"x":[],"y":[],"angle":[],"front":[],"right":[],"left":[],"dim":p.model.indim}
    state = p.reset()
    while not p.done:
        # print every state and position of the car
        point = p.car.getPosition('center')

        # select action randomly
        # you can predict your action according to the state here
        angle = p.predictAction(state)[0] # 0~80
        
        if angle < 0:
            angle = 0
        if angle > 80:
            angle = 80

        path['x'].append(point.x)
        path['y'].append(point.y)
        path['front'].append(state[0])
        path['right'].append(state[1])
        path['left'].append(state[2])
        path['angle'].append(angle-40)
        
        
        # smooth_range = min(len(angles),3)
        # smoothed = smooth(angles[::-1][:smooth_range], smooth_range)[0]
        # smoothed = pre_angle*0.1 + angle*0.9
        # smoothed = angle
        # print(f"x1:{state[0]:07.3f}, x2:{state[1]:07.3f}, x3:{state[2]:07.3f}, diff:{state[2]-state[1]:07.3f}, angle: {angle-40:07.3f}, smooth: {smoothed-40:07.3f}")
        # take action
        state = p.step(angle)
    record_path(path)
    return path

def record_path(path:dict):
    if path["dim"] == 4:
        generator = zip(path["x"],path["y"],path["front"],path["right"],path["left"],path["angle"])
    else:
        generator = zip(path["front"],path["right"],path["left"],path["angle"])
    with open(f"./track{path['dim']+2}D.txt","w") as f:
        for tup_data in generator:
            line = " ".join([f"{i:-012.8f}" for i in tup_data]) + "\n"
            f.write(line)



def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

def smooth(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == "__main__":
    # data = np.loadtxt("./train4dAll.txt")
    # new_data = np.stack([data[:,0],data[:,2]-data[:,1]]).T
    # rbf = RBF(2, 800, 1)
    # data[:,0] = scale_range(data[:,0], min(data[:,0])+2, max(data[:,0])-2)
    # data[:,1] = scale_range(data[:,1], min(data[:,1])+2, max(data[:,1])-2)
    # data[:,2] = scale_range(data[:,2], min(data[:,2])+2, max(data[:,2])-2)
    # rbf.train(new_data, data[:,3])

    data = np.loadtxt("./train6dAll.txt")
    new_data = np.stack([data[:,0],data[:,1],data[:,2],data[:,4]-data[:,3]]).T
    rbf = RBF(4, 1000, 1)
    rbf.train(new_data, data[:,5])
    
    p = Playground(rbf)

    path = run_example(p)
    plt.cla()
    plt.plot(path['x'],path['y'],alpha=0.8) # draw car go path
    for line in p.lines: # draw road
        plt.plot([line.p1.x,line.p2.x],[line.p1.y,line.p2.y],c='r')
    plt.show()
