using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Agents.PPO.Implementations;
using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RLMatrix.Agents.Common
{
    /// <summary>
    /// Represents a local discrete rollout agent for reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type of the state.</typeparam>
    public partial class LocalDiscreteRolloutAgent<TState> : IDiscreteRolloutAgent<TState>
    {
        protected readonly Dictionary<Guid, IEnvironmentAsync<TState>> _environments;
        protected readonly Dictionary<Guid, Episode<TState>> _ennvPairs;
        protected readonly IDiscreteProxy<TState> _agent;
        private readonly RLMatrix.Agents.DQN.Domain.IBehavioralCloning<TState>? _behavioralCloning;
        /// <summary>
        /// Initializes a new instance of the <see cref="LocalDiscreteRolloutAgent{TState}"/> class with DQN agent options.
        /// </summary>
        /// <param name="options">The DQN agent options.</param>
        /// <param name="environments">The collection of environments.</param>
        public LocalDiscreteRolloutAgent(DQNAgentOptions options, IEnumerable<IEnvironmentAsync<TState>> environments)
        {
            _environments = environments.ToDictionary(env => Guid.NewGuid(), env => env);
            _ennvPairs = _environments.ToDictionary(pair => pair.Key, pair => new Episode<TState>());
            _agent = new LocalDiscreteQAgent<TState>(options, environments.First().actionSize, environments.First().stateSize);
        }

        // Backwards-compatible Step overload for projects expecting the original signature
        public Task Step(bool isTraining = true)
        {
            return Step(isTraining, false);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LocalDiscreteRolloutAgent{TState}"/> class with PPO agent options.
        /// </summary>
        /// <param name="options">The PPO agent options.</param>
        /// <param name="environments">The collection of environments.</param>
        public LocalDiscreteRolloutAgent(PPOAgentOptions options, IEnumerable<IEnvironmentAsync<TState>> environments, RLMatrix.Agents.DQN.Domain.IBehavioralCloning<TState>? behavioralCloning = null)
        {
            _environments = environments.ToDictionary(env => Guid.NewGuid(), env => env);
            _ennvPairs = _environments.ToDictionary(pair => pair.Key, pair => new Episode<TState>());
            _agent = new LocalDiscretePPOAgent<TState>(options, environments.First().actionSize, environments.First().stateSize, behavioralCloning);
            _behavioralCloning = behavioralCloning;
        }


        /// <summary>
        /// Performs a step in the reinforcement learning process.
        /// </summary>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task Step(bool isTraining = true, bool behaviorCloning = false)
        {
            List<Task<(Guid environmentId, TState state)>> stateTaskList = new List<Task<(Guid environmentId, TState state)>>();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                stateTaskList.Add(stateTask);
            }
            var stateResults = await Task.WhenAll(stateTaskList);

            List<(Guid environmentId, TState state)> payload = stateResults.ToList();
            Dictionary<Guid, int[]> actions;

            if (behaviorCloning && _behavioralCloning != null)
            {
                actions = new Dictionary<Guid, int[]>();
                // For each environment, attempt to call an expert action method decorated with RLMatrixExpertActionDiscreteAttribute
                foreach (var (envId, state) in payload)
                {
                    var env = _environments[envId];
                    var envObj = env;
                        var method = envObj.GetType().GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                        .FirstOrDefault(m => m.GetCustomAttributes(true).Any(a => a.GetType().Name == "RLMatrixExpertActionDiscreteAttribute"));

                    if (method != null)
                    {
                        object? result = null;
                        var parameters = method.GetParameters();
                        if (parameters.Length == 0)
                        {
                            result = method.Invoke(envObj, null);
                        }
                        else if (parameters.Length == 1 && parameters[0].ParameterType.IsAssignableFrom(state?.GetType()))
                        {
                            result = method.Invoke(envObj, new object[] { state });
                        }

                        if (result is int single)
                        {
                            actions[envId] = new int[] { single };
                        }
                        else if (result is int[] arr)
                        {
                            actions[envId] = arr;
                        }
                        else
                        {
                            // fallback to random action if expert method returned unexpected type
                            actions[envId] = Enumerable.Range(0, env.actionSize.Length).Select(i => 0).ToArray();
                        }
                    }
                    else
                    {
                        // no expert method, fallback to agent
                        // call agent select for this single state
                        var single = await _agent.SelectActionsBatchAsync(new List<(Guid, TState)> { (envId, state) }, isTraining);
                        actions[envId] = single[envId];
                    }
                }
            }
            else
            {
                var selected = await _agent.SelectActionsBatchAsync(payload, isTraining);
                actions = selected;
            }

            List<Task<(Guid environmentId, (float, bool) reward)>> rewardTaskList = new List<Task<(Guid environmentId, (float, bool) reward)>>();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var rewardTask = env.Step(action.Value)
                    .ContinueWith(t => (action.Key, t.Result));
                rewardTaskList.Add(rewardTask);
            }

            await Task.WhenAll(rewardTaskList);

            List<Task<(Guid environmentId, TState state)>> nextStateTaskList = new List<Task<(Guid environmentId, TState state)>>();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                nextStateTaskList.Add(stateTask);
            }
            var nextStateResults = await Task.WhenAll(nextStateTaskList);

            ConcurrentBag<TransitionPortable<TState>> transitionsToShip = new ConcurrentBag<TransitionPortable<TState>>();
            ConcurrentBag<double> rewards = new ConcurrentBag<double>();
            var completedEpisodes = new List<(Guid environmentId, bool done)>();

            foreach (var env in _environments)
            {
                var key = env.Key;
                var episode = _ennvPairs[key];
                var stateResult = stateResults.First(x => x.environmentId == key);
                var state = stateResult.state;
                var action = actions[key];
                var stepResult = rewardTaskList.First(x => x.Result.environmentId == key).Result;
                var reward = stepResult.reward.Item1;
                var isDone = stepResult.reward.Item2;
                var nextState = nextStateResults.First(x => x.environmentId == key).state;
                episode.AddTransition(state, isDone, action, null, reward);
                if (isDone)
                {
                    foreach (var transition in episode.CompletedEpisodes)
                    {
                        transitionsToShip.Add(transition);
                        // If behavior cloning is active, also add to expert demonstrations memory
                        if (behaviorCloning && _behavioralCloning != null)
                        {
                            var inMem = episode.CompletedEpisodes.ToTransitionInMemory();
                            _behavioralCloning.ExpertDemonstrations.Push(inMem);
                            break; // we've pushed the episode, avoid pushing repeatedly per transition
                        }
                    }
                    rewards.Add(episode.cumulativeReward);
                    episode.CompletedEpisodes.Clear();
                    completedEpisodes.Add((key, true));
                }
            }


            if (transitionsToShip.Count > 0)
            {
                if (isTraining)
                {
                    await _agent.UploadTransitionsAsync(transitionsToShip.ToList());
                }
                // Always clear transitions regardless of training mode
            }
            // Always clear transitions regardless of training mode
            #if NET8_0_OR_GREATER
            transitionsToShip.Clear();
            #else
            transitionsToShip = new ConcurrentBag<TransitionPortable<TState>>();
            #endif

            await _agent.ResetStates(completedEpisodes);

            if (!isTraining)
            {
                // Periodic cleanup to prevent memory leaks in evaluation mode
                if (_agent is IHasMemory<TState> memoryAgent &&
                    memoryAgent.Memory.Length > 10000)
                {
                    memoryAgent.Memory.ClearMemory();
                }
                return;
            }

            await _agent.OptimizeModelAsync();
        }

        public void StepSync(bool isTraining = true)
        {
            List<(Guid environmentId, TState state)> stateResults = new List<(Guid environmentId, TState state)>();
            foreach (var env in _environments)
            {
                var state = GetStateSync(env.Key, env.Value);
                stateResults.Add(state);
            }

            List<(Guid environmentId, TState state)> payload = stateResults;
            var actions = GetActionsBatchSync(payload, isTraining);

            List<(Guid environmentId, (float, bool) reward)> rewardResults = new List<(Guid environmentId, (float, bool) reward)>();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var reward = env.Step(action.Value).GetAwaiter().GetResult();
                rewardResults.Add((action.Key, reward));
            }

            List<(Guid environmentId, TState state)> nextStateResults = new List<(Guid environmentId, TState state)>();
            foreach (var env in _environments)
            {
                var state = GetStateSync(env.Key, env.Value);
                nextStateResults.Add(state);
            }

            var transitionsToShip = new List<TransitionPortable<TState>>();
            var rewards = new List<double>();
            var completedEpisodes = new List<(Guid environmentId, bool done)>();

            foreach (var env in _environments)
            {
                var key = env.Key;
                var episode = _ennvPairs[key];
                var stateResult = stateResults.First(x => x.environmentId == key);
                var state = stateResult.state;
                var action = actions[key];
                var stepResult = rewardResults.First(x => x.environmentId == key);
                var reward = stepResult.Item2.Item1;
                var isDone = stepResult.Item2.Item2;
                var nextState = nextStateResults.First(x => x.environmentId == key).state;
                episode.AddTransition(state, isDone, action, null, reward);
                if (isDone)
                {
                    transitionsToShip.AddRange(episode.CompletedEpisodes);
                    rewards.Add(episode.cumulativeReward);
                    episode.CompletedEpisodes.Clear();
                    completedEpisodes.Add((key, true));
                }
            }

            if (transitionsToShip.Count > 0)
            {
                _agent.UploadTransitionsAsync(transitionsToShip).GetAwaiter().GetResult();
            }

            _agent.ResetStates(completedEpisodes).GetAwaiter().GetResult();

            if (isTraining)
            {
                _agent.OptimizeModelAsync().GetAwaiter().GetResult();
            }
        }

        private (Guid environmentId, TState state) GetStateSync(Guid environmentId, IEnvironmentAsync<TState> env)
        {
            var state = DeepCopy(env.GetCurrentState().GetAwaiter().GetResult());
            return (environmentId, state);
        }

        private Dictionary<Guid, int[]> GetActionsBatchSync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
        {
            return _agent.SelectActionsBatchAsync(stateInfos, isTraining).GetAwaiter().GetResult();
        }


        /// <summary>
        /// Gets actions for a batch of states asynchronously.
        /// </summary>
        /// <param name="stateInfos">The list of state information.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>A dictionary of actions for each environment.</returns>
#if NET8_0_OR_GREATER
        public ValueTask<Dictionary<Guid, int[]>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
        {
            return _agent.SelectActionsBatchAsync(stateInfos, isTraining);
        }
#else
    public Task<Dictionary<Guid, int[]>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
    {
        return _agent.SelectActionsBatchAsync(stateInfos, isTraining);
    }
#endif

        private TState DeepCopy(TState input)
        {
            if (input is float[] array1D)
            {
                return (TState)(object)array1D.ToArray();
            }
            else if (input is float[,] array2D)
            {
                int rows = array2D.GetLength(0);
                int cols = array2D.GetLength(1);
                var clone = new float[rows, cols];
                Buffer.BlockCopy(array2D, 0, clone, 0, array2D.Length * sizeof(float));
                return (TState)(object)clone;
            }
            else
            {
                throw new InvalidOperationException("This method can only be used with float[] or float[,].");
            }
        }

        private async Task<(Guid environmentId, TState state)> GetStateAsync(Guid environmentId, IEnvironmentAsync<TState> env)
        {
            var state = DeepCopy(await env.GetCurrentState());
            return (environmentId, state);
        }

        /// <summary>
        /// Saves the agent's state to the specified path.
        /// </summary>
        /// <param name="path">The path to save the agent's state.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
#if NET8_0_OR_GREATER
        public ValueTask Save(string path)
        {
            return _agent.SaveAsync(path);
        }
#else
        public Task Save(string path)
        {
            return _agent.SaveAsync(path);
            return Task.CompletedTask;
        }
#endif

        /// <summary>
        /// Loads the agent's state from the specified path.
        /// </summary>
        /// <param name="path">The path to load the agent's state from.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
#if NET8_0_OR_GREATER
        public ValueTask Load(string path)
        {
            return _agent.LoadAsync(path);
        }
#else
        public Task Load(string path)
        {
            return _agent.LoadAsync(path);
            return Task.CompletedTask;
        }
#endif
    }
}