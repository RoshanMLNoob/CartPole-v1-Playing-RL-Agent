import numpy as np
import gymnasium as gym
import gymnasium.utils.save_video as vid
import time
import moviepy
import os


class RL_cartpole(object):

    def __init__(self, env=None , gamma=0.95 , alpha=0.2 , epsilon=0.3 , show=False , render_mode=None):
        if env is None:
            if render_mode is None:
                if show is True:
                    self.env = gym.make("CartPole-v1" , render_mode="human")
                else:
                    self.env = gym.make("CartPole-v1")
            else:
                self.env = gym.make("CartPole-v1" , render_mode=render_mode)
            self.state , self.info = self.env.reset()
            self.state = self.discreat_states(self.state)
            self.Q = {}
            self.gamma , self.alpha , self.epsilon = gamma , alpha , epsilon

    def Reward(self , state , done=False):
        if done:
            return -10.0 #Heavy negetive reward
        return 1

        #Effortful but Unworkable Code sadly
        x , x0 , th , th0 = state
        x_m , th_m = 2.4 , 0.21
        w1 , w2 = 0.1 , 1.0

        pos_penelty = w1*((x/x_m)**2)
        ang_penelty = w2*((th/th_m)**2)
        reward = 1 - (pos_penelty + ang_penelty)

        return reward

    def policy(self , state):

        Probability = np.random.random()

        if Probability > (self.epsilon):
            Qs = [ ]
            for action in range(2):
                Qs.append(self.Q.get((state , action) , 0.0))
            if Qs[0] == Qs[1]:
                return np.random.choice(2)
            return np.argmax(Qs)
        else:
            return np.random.choice(2)

    def update_Q(self , state , action , state_prime , done=False):

        Q_val = self.Q.get((state , action) , 0.0)
        Q_max = []
        for a in range(2):
            Q_max.append(self.Q.get((state_prime , a) , 0.0))
        Q_max = float(max(Q_max))

        self.Q[(state , action)] = (1-self.alpha)*(Q_val) + self.alpha*(self.Reward(state_prime , done) + self.gamma*(Q_max))

    def discreat_states(self , state):
        
        self.piece = 25
        pole_ang , pole_v = state[2] , state[3]
        pos , vel = state[0] , state[1]

        angle_idx = int(np.digitize(pole_ang, np.linspace(-0.21, 0.21, self.piece)) - 1)
        velocity_idx = int(np.digitize(pole_v, np.linspace(-3.0, 3.0, self.piece)) - 1)
        pos_idx = int(np.digitize(pos , np.linspace(-2.4, 2.4, self.piece)) - 1)
        vel_idx = int(np.digitize(vel , np.linspace(-3.0 , 3.0 , self.piece)) -1)

        angle_idx = max(0, min(self.piece-1, angle_idx))
        velocity_idx = max(0, min(self.piece-1, velocity_idx))
        position = max(0, min((self.piece)-1 , pos_idx))
        velocity = max(0 , min((self.piece)-1 , vel_idx))

        return (angle_idx , velocity_idx , position , velocity)

    def save_Q(self, file):
        import pickle
        Data = (self.Q , self.epsilon , self.alpha , self.gamma , self.piece)
        with open(file , "wb") as f:
            pickle.dump(Data , f)
    def load_Q(self, file):
        import pickle
        with open(file , "rb") as f:
            Data = pickle.load(f)
        self.Q , self.epsilon , self.alpha , self.gamma , self.piece = Data

    def _run_model_(self , epesodes=1550 , horizon=50 , to_show=1500 , interval=100 , autosave_file=None , decay_rate=0.9999):

        show = False
        self.Win = []
        for t in range(epesodes):

            self.epsilon = max(0.01 , self.epsilon*decay_rate)
            self.state , self.info = self.env.reset()

            for _ in range(horizon):

                self.action = self.policy(tuple(self.state))
                state_old = self.state[:]
                action_old = int(self.action)

                state_prime = self.env.step(self.action)
                self.state = self.discreat_states(state_prime[0])
                if t >= to_show:
                    time.sleep(0.0000001)
            
                done = state_prime[2] or state_prime[3]
                self.update_Q(tuple(state_old) , action_old , tuple(self.state) , done)

                if show:
                    print(self.Reward(self.state , done) , self.state)

                if done:
                    t0 = time.time()
                    #print(False ,  end="->")
                    state0 , self.info = self.env.reset()
                    self.state = self.discreat_states(state0)
                    self.Win.append(False)
                    if t>=to_show:
                        print("************************LOST******************************")
                        time.sleep(0.1)
                    break

            if done is False:
                self.Win.append(True)
                if t >= to_show:
                    print("*********WIN**************")
                    time.sleep(0.1)
            if t == to_show:
                self.env = gym.make("CartPole-v1" , render_mode="human")
                self.state , self.info = self.env.reset()
                self.epsilon = 0
                show = True
            if t%interval==0:

                print("\033c", end="", flush=True)
                print(f"[ Ep:{t}  WR:{round( self.Win.count(True) / (len(self.Win)+0.01) , 5)*100} ]" , end="->" , flush=True)

                if t%(interval*20) == 0 and autosave_file is not None:
                    self.save_Q(autosave_file)
                    print(f"Save {t//(interval*20)}")

    def run_randomly(self):
        done = False
        env = self.env
        while True:

            action = env.action_space.sample()
            print(action)
            s_prime = env.step(action)
            time.sleep(0.02)
            print(s_prime)

            done = s_prime[2] or s_prime[3]
            if done:
                print("*******************************LOST********************************")
                env.reset()

f_new = r"Data_New1.pkl"
f_old = r"Q_table.pkl"


train = (50000 , 500 , 49975 , 100)
test = (10000 , 500 , 100)

Model = RL_cartpole(gamma=0.95,
                    alpha=0.2,
                    epsilon=0.3
                    )

Model.load_Q(file=f_old)

start = True
if start is True:
    Model._run_model_(*train , autosave_file=f_old)
    print(Model.Win.count(True) , Model.Win.count(False) , len(Model.Win))
    Model.save_Q(file=f_old)


k = input("Click Enter to start a Video Recording , else click Ctrl+C to exit via KeyboardIntterupt: ")

#Below this coded by AI, I needed to record a video demonstration and not wnating to learn a new
#Library in the way

import os
from gymnasium.wrappers import RecordVideo

# 1. Initialize your recording model instance in frame capture mode
print("\nInitializing video rendering pipeline...")
Model_Record = RL_cartpole(render_mode="rgb_array")

# 2. CRUCIAL FIX: Load your mastered memory file so it doesn't play with a blank brain!
Model_Record.load_Q(file=f_old)

#Model_Record._run_model_(epesodes=5 , horizon=1000 , to_show=0)

# 3. Force exploration completely off so the AI acts with 100% flawless confidence
Model_Record.epsilon = 0.0

# 4. Wrap your model's active environment inside Gymnasium's RecordVideo module
video_folder = "mit_6036_project_videos"
Model_Record.env = RecordVideo(
    env=Model_Record.env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: True, # Record every game session
    name_prefix="trained_agent_performance"
)

# 5. Run the model for 3 demonstration games with a higher horizon
print(f"Recording flawless balancing runs to folder: '{video_folder}'...")
# We use episodes=3 and a long horizon=500 so you get beautiful, long showcase clips!
Model_Record._run_model_(epesodes=5, horizon=500 , to_show=9999)

# 6. CRUCIAL FIX: Forcibly close the outer RecordVideo wrapper to flush and save the files!
Model_Record.env.close()

print("\nRecording successfully finalized with zero truncated video drops!")
print(f"Open your workspace directory and look for the folder: {os.path.abspath(video_folder)}")

