using RLMatrix.Agents.Common;
using RLMatrix.Memories;
using RLMatrix.Agents.DQN.Domain;
using System;
using TorchSharp;
using static TorchSharp.torch;

namespace RLMatrix.Imitation
{
    public class BehavioralCloning<T> : IBehavioralCloning<T>
    {
        private readonly IMemory<T> expertDemonstrations;
        private readonly int epochs;
        private readonly int batchSize;

        public BehavioralCloning(IMemory<T> expertDemonstrations, int epochs = 3, int batchSize = 64)
        {
            this.expertDemonstrations = expertDemonstrations ?? throw new ArgumentNullException(nameof(expertDemonstrations));
            this.epochs = Math.Max(1, epochs);
            this.batchSize = Math.Max(1, batchSize);
        }

        public IMemory<T> ExpertDemonstrations => expertDemonstrations;

        public void TrainActor(RLMatrix.PPOActorNet actorNet, TorchSharp.torch.Device device, int[] actionSizes, (float, float)[] continuousActionBounds, PPOAgentOptions options)
        {
            if (expertDemonstrations == null || expertDemonstrations.Length == 0)
                return;

            // Simple behavioral cloning training loop: maximize log-probability of expert actions
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // sample a minibatch (or entire memory if smaller)
                var batchSizeToUse = Math.Min(batchSize, expertDemonstrations.Length);
                var transitions = expertDemonstrations.Sample(batchSizeToUse);

                // Create tensors from transitions similar to PPO's helper
                int length = transitions.Count;
                transitions.UnpackTransitionInMemory(out T[] batchStates, out int[,] batchDiscreteActions, out float[,] batchContinuousActions);

                var stateBatch = RLMatrix.Agents.Common.Utilities<T>.StateBatchToTensor(batchStates, device);

                Tensor actionBatch;
                using (var discreteActionBatch = tensor(batchDiscreteActions, dtype: torch.int64, device: device))
                using (var continuousActionBatch = tensor(batchContinuousActions, device: device))
                {
                    actionBatch = torch.cat(new Tensor[] { discreteActionBatch, continuousActionBatch }, dim: 1);
                }

                using (var actorScope = torch.NewDisposeScope())
                {
                    // actorNet is expected to implement get_log_prob similar to PPOActorNet
                    var ppoActor = actorNet;
                    var logProbs = ppoActor.get_log_prob(stateBatch, actionBatch, actionSizes.Length, continuousActionBounds.Length);
                    var loss = -logProbs.mean();

                    // create a local optimizer for behavioral cloning (do not depend on OptimizerHelper)
                    using (var opt = optim.Adam(ppoActor.parameters(), lr: options.LR))
                    {
                        opt.zero_grad();
                        loss.backward();
                        torch.nn.utils.clip_grad_norm_(actorNet.parameters(), options.ClipGradNorm);
                        opt.step();
                    }
                }
            }
        }
    }
}
