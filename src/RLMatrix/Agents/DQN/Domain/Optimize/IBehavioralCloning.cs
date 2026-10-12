using RLMatrix.Memories;
using TorchSharp;

namespace RLMatrix.Agents.DQN.Domain
{
    public interface IBehavioralCloning<T>
    {
        /// <summary>
        /// Train the actor network using expert demonstrations stored inside the implementation.
        /// The actor net type is PPOActorNet to match PPO implementations.
        /// </summary>
        void TrainActor(RLMatrix.PPOActorNet actorNet, TorchSharp.torch.Device device, int[] actionSizes, (float, float)[] continuousActionBounds, PPOAgentOptions options);
    }
}
