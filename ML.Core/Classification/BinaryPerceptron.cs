using ML.Core.Models;
using ML.Core.Preprocessing;

namespace ML.Core.Classification;

public sealed class BinaryPerceptron : IProgressiveClassifier
{
    private readonly Random _random;
    private readonly ZScoreScaler _scaler = new();
    private double[] _weights = Array.Empty<double>();
    private double _bias;

    public double LearningRate { get; }
    public int MaxEpochs { get; }
    public double Momentum { get; }

    public BinaryPerceptron(double learningRate = 0.001, int maxEpochs = 10_000, double momentum = 0.9, int seed = 42)
    {
        if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (maxEpochs <= 0) throw new ArgumentOutOfRangeException(nameof(maxEpochs));
        if (momentum is < 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(momentum));
        LearningRate = learningRate;
        MaxEpochs = maxEpochs;
        Momentum = momentum;
        _random = new Random(seed);
    }

    public TrainingResult Fit(IReadOnlyList<double[]> samples, IReadOnlyList<int> labels) =>
        FitWithProgress(samples, labels, null, 1);

    public TrainingResult FitWithProgress(
        IReadOnlyList<double[]> samples,
        IReadOnlyList<int> labels,
        Action<ClassifierEpochSnapshot>? onSnapshot,
        int snapshotEvery = 1)
    {
        int dim = ClassifierValidation.Validate(samples, labels);
        if (labels.Any(y => y is not (0 or 1)))
            throw new ArgumentException("Binary Perceptron yalnızca 0 ve 1 sınıflarını destekler.", nameof(labels));

        snapshotEvery = Math.Max(1, snapshotEvery);
        _scaler.Fit(samples);
        var x = _scaler.Transform(samples);
        _weights = Enumerable.Range(0, dim).Select(_ => (_random.NextDouble() - 0.5) * 0.1).ToArray();
        _bias = 0.0;
        var previousWeightUpdates = new double[dim];
        double previousBiasUpdate = 0.0;
        var history = new List<double>(MaxEpochs);

        int epoch;
        for (epoch = 1; epoch <= MaxEpochs; epoch++)
        {
            int mistakes = 0;
            for (int k = 0; k < x.Length; k++)
            {
                int target = labels[k] == 1 ? 1 : -1;
                double net = _bias;
                for (int d = 0; d < dim; d++) net += _weights[d] * x[k][d];
                int output = net >= 0 ? 1 : -1;
                int error = target - output;
                if (error == 0) continue;

                mistakes++;
                for (int d = 0; d < dim; d++)
                {
                    double current = LearningRate * (error / 2.0) * x[k][d];
                    double update = current + Momentum * previousWeightUpdates[d];
                    _weights[d] += update;
                    previousWeightUpdates[d] = update;
                }

                double biasCurrent = LearningRate * (error / 2.0);
                double biasUpdate = biasCurrent + Momentum * previousBiasUpdate;
                _bias += biasUpdate;
                previousBiasUpdate = biasUpdate;
            }

            history.Add(mistakes);
            if (onSnapshot is not null && (epoch <= 12 || epoch % snapshotEvery == 0 || mistakes == 0))
                onSnapshot(new ClassifierEpochSnapshot { Epoch = epoch, Loss = mistakes, Model = CreateSnapshot() });
            if (mistakes == 0) break;
        }

        int epochsRun = Math.Min(epoch, MaxEpochs);
        return new TrainingResult
        {
            EpochsRun = epochsRun,
            FinalLoss = history.Count == 0 ? 0 : history[^1],
            Accuracy = ClassifierValidation.Accuracy(this, samples, labels),
            ErrorHistory = history
        };
    }

    public int Predict(IReadOnlyList<double> sample)
    {
        EnsureTrained();
        var x = _scaler.Transform(sample);
        double net = _bias;
        for (int d = 0; d < _weights.Length; d++) net += _weights[d] * x[d];
        return net >= 0 ? 1 : 0;
    }

    public double[] PredictScores(IReadOnlyList<double> sample)
    {
        EnsureTrained();
        var x = _scaler.Transform(sample);
        double net = _bias;
        for (int d = 0; d < _weights.Length; d++) net += _weights[d] * x[d];
        double p1 = 1.0 / (1.0 + Math.Exp(-Math.Clamp(net, -40, 40)));
        return [1.0 - p1, p1];
    }

    private IClassifier CreateSnapshot()
    {
        var weights = _weights.ToArray();
        double bias = _bias;
        var mean = _scaler.Mean.ToArray();
        var std = _scaler.Std.ToArray();
        return new FrozenClassifier(sample =>
        {
            double net = bias;
            for (int i = 0; i < weights.Length; i++)
            {
                double z = (sample[i] - mean[i]) / std[i];
                net += weights[i] * z;
            }
            double p1 = 1.0 / (1.0 + Math.Exp(-Math.Clamp(net, -40, 40)));
            return [1.0 - p1, p1];
        });
    }

    private void EnsureTrained()
    {
        if (_weights.Length == 0) throw new InvalidOperationException("Model henüz eğitilmedi.");
    }
}
