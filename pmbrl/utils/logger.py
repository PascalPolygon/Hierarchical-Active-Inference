import os
import json
from datetime import datetime
import pprint


class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = "log_" + logdir + "_" + str(seed) + "/"
        self.print_path = self.path + "out.txt"
        self.metrics_path = self.path + "metrics.json"
        self.video_dir = self.path + "videos/"
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self.metrics = {}
        self.trial_metrics = {}
        self._init_print()
        self._setup_metrics()

    def log(self, string):
        """
        Logs a string to the output file and prints it.

        Args:
            string (str): The message to log.
        """
        with open(self.print_path, "a") as f:
            f.write("\n")
            f.write(str(string))
        print(string)

    def log_losses(self, e_loss, r_loss):
        """
        Logs ensemble and reward losses.

        Args:
            e_loss (float): Ensemble loss.
            r_loss (float): Reward loss.
        """
        self.metrics["e_losses"].append(e_loss)
        self.metrics["r_losses"].append(r_loss)
        msg = "Ensemble Loss: {:.2f} | Reward Loss: {:.2f}"
        self.log(msg.format(e_loss, r_loss))

    def log_coverage(self, coverage):
        """
        Logs coverage statistics.

        Args:
            coverage (float): Coverage metric.
        """
        self.metrics["coverage"].append(coverage)
        msg = "Coverage: {:.2f}"
        self.log(msg.format(coverage))

    def log_episode(self, reward, steps):
        """
        Logs episode rewards and steps.

        Args:
            reward (float): Total reward for the episode.
            steps (int): Total steps taken in the episode.
        """
        self.metrics["rewards"].append(reward)
        self.metrics["steps"].append(steps)
        msg = "Episode Reward: {:.2f} | Steps: {:.2f}"
        self.log(msg.format(reward, steps))
    
    def log_trial_episode(self, trial_num, episode, reward, steps, stats, agent_name):
        if agent_name not in self.trial_metrics:
            self.trial_metrics[agent_name] = {}
        if trial_num not in self.trial_metrics[agent_name]:
            self.trial_metrics[agent_name][trial_num] = []
        self.trial_metrics[agent_name][trial_num].append({
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'stats': stats,
        })

    def log_time(self, time_taken):
        """
        Logs the time taken for an episode.

        Args:
            time_taken (float): Time taken in seconds.
        """
        self.metrics["times"].append(time_taken)
        self.log("Episode Time: {:.2f} seconds".format(time_taken))

    def log_stats(self, stats):
        """
        Logs various statistics from the planner.

        Args:
            stats (dict): A dictionary containing various statistics.
                          Expected keys include 'reward', 'exploration',
                          'goal_achievement', 'goal_reward', 'goal_exploration'.
        """
        for key, value in stats.items():
            # Map the planner's keys to logger's metrics keys
            metric_key = f"{key}_stats"

            # Initialize the metric list if it doesn't exist
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []

            # Append the current stats to the metrics
            self.metrics[metric_key].append(value)

            # Format the values for logging
            formatted_stats = {k: "{:.2f}".format(v) for k, v in value.items()}

            # Determine the appropriate log message based on the key
            if key == "reward":
                self.log("Reward Stats:\n {}".format(pprint.pformat(formatted_stats)))
            elif key == "exploration":
                self.log(
                    "Exploration Stats:\n {}".format(pprint.pformat(formatted_stats))
                )
            elif key == "goal_achievement":
                self.log(
                    "Goal Achievement Stats:\n {}".format(
                        pprint.pformat(formatted_stats)
                    )
                )
            elif key == "goal_reward":
                self.log(
                    "Goal Reward Stats:\n {}".format(pprint.pformat(formatted_stats))
                )
            elif key == "goal_exploration":
                self.log(
                    "Goal Exploration Stats:\n {}".format(
                        pprint.pformat(formatted_stats)
                    )
                )
            else:
                # For any unforeseen keys, log them generically
                self.log(
                    f"{key.capitalize()} Stats:\n {pprint.pformat(formatted_stats)}"
                )

    # def save(self):
    #     """
    #     Saves all logged metrics to a JSON file.
    #     """
    #     self._save_json(self.metrics_path, self.metrics)
    #     self.log("Metrics saved to {}".format(self.metrics_path))
    
    def save(self):
        # Save metrics to JSON file
        os.makedirs(self.logdir, exist_ok=True)
        metrics_path = os.path.join(self.logdir, 'metrics.json')
        self.log("Metrics saved to {}".format(metrics_path))
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def get_video_path(self, episode):
        """
        Generates the video file path for a given episode.

        Args:
            episode (int): Episode number.

        Returns:
            str: Path to the video file.
        """
        return os.path.join(self.video_dir, f"{episode}.mp4")

    def _init_print(self):
        """
        Initializes the print log with the current time.
        """
        with open(self.print_path, "w") as f:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Logging started at {current_time}\n")

    def _setup_metrics(self):
        """
        Initializes the metrics dictionary with empty lists for each metric.
        """
        self.metrics = {
            "e_losses": [],
            "r_losses": [],
            "rewards": [],
            "steps": [],
            "times": [],
            "reward_stats": [],
            "info_stats": [],
            "coverage": [],
            "goal_achievement_stats": [],  # New metric
            "goal_reward_stats": [],  # New metric
            "goal_exploration_stats": [],  # New metric
        }

    def _save_json(self, path, obj):
        """
        Saves a dictionary to a JSON file.

        Args:
            path (str): Path to the JSON file.
            obj (dict): Dictionary to save.
        """
        with open(path, "w") as file:
            json.dump(obj, file, indent=4)
    
    def save_metrics(self, all_metrics, trial):
        metrics_path = os.path.join(self.logdir, f'metrics_trial_{trial}.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
