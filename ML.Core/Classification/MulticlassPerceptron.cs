using ML.Core.Models;
using ML.Core.Preprocessing;

namespace ML.Core.Classification;

public sealed class MulticlassPerceptron : IProgressiveClassifier
{
    private readonly Random _random;
    private readonly ZScoreScaler _scaler = new();
    private double[][] _weights = Array.Empty<double[]>();
    private double[] _biases = Array.Empty<double>();

    public double LearningRate { get; }
    public int MaxEpochs { get; }
    public double Momentum { get; }
    public int ClassCount => _weights.Length;

    public MulticlassPerceptron(double learningRate = 0.001, int maxEpochs = 10_000, double momentum = 0.9, int seed = 42)
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
        int classCount = labels.Max() + 1;
        snapshotEvery = Math.Max(1, snapshotEvery);
        _scaler.Fit(samples);
        var x = _scaler.Transform(samples);

        _weights = new double[classCount][];
        _biases = new double[classCount];
        var previousWeightUpdates = new double[classCount][];
        var previousBiasUpdates = new double[classCount];
        for (int c = 0; c < classCount; c++)
        {
            _weights[c] = Enumerable.Range(0, dim).Select(_ => (_random.NextDouble() - 0.5) * 0.1).ToArray();
            previousWeightUpdates[c] = new double[dim];
        }

        var history = new List<double>(MaxEpochs);
        int epoch;
        for (epoch = 1; epoch <= MaxEpochs; epoch++)
        {
            int mistakes = 0;
            for (int k = 0; k < x.Length; k++)
            {
                int predicted = PredictNormalized(x[k]);
                int expected = labels[k];
                if (predicted == expected) continue;
                mistakes++;

                for (int d = 0; d < dim; d++)
                {
                    double wrongCurrent = -LearningRate * x[k][d];
                    double wrongUpdate = wrongCurrent + Momentum * previousWeightUpdates[predicted][d];
                    _weights[predicted][d] += wrongUpdate;
                    previousWeightUpdates[predicted][d] = wrongUpdate;

                    double rightCurrent = LearningRate * x[k][d];
                    double rightUpdate = rightCurrent + Momentum * previousWeightUpdates[expected][d];
                    _weights[expected][d] += rightUpdate;
                    previousWeightUpdates[expected][d] = rightUpdate;
                }

                double wrongBiasUpdate = -LearningRate + Momentum * previousBiasUpdates[predicted];
                _biases[predicted] += wrongBiasUpdate;
                previousBiasUpdates[predicted] = wrongBiasUpdate;
                double rightBiasUpdate = LearningRate + Momentum * previousBiasUpdates[expected];
                _biases[expected] += rightBiasUpdate;
                previousBiasUpdates[expected] = rightBiasUpdate;
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
        return PredictNormalized(_scaler.Transform(sample));
    }

    public double[] PredictScores(IReadOnlyList<double> sample)
    {
        EnsureTrained();
        var scores = RawScores(_scaler.Transform(sample));
        double max = scores.Max();
        var exp = scores.Select(v => Math.Exp(v - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    private int PredictNormalized(double[] x)
    {
        var scores = RawScores(x);
        int best = 0;
        for (int c = 1; c < scores.Length; c++) if (scores[c] > scores[best]) best = c;
        return best;
    }

    private double[] RawScores(double[] x)
    {
        var scores = new double[_weights.Length];
        for (int c = 0; c < _weights.Length; c++)
        {
            double score = _biases[c];
            for (int d = 0; d < x.Length; d++) score += _weights[c][d] * x[d];
            scores[c] = score;
        }
        return scores;
    }

    private IClassifier CreateSnapshot()
    {
        var weights = _weights.Select(row => row.ToArray()).ToArray();
        var biases = _biases.ToArray();
        var mean = _scaler.Mean.ToArray();
        var std = _scaler.Std.ToArray();
        return new FrozenClassifier(sample =>
        {
            var exp = new double[weights.Length];
            var raw = new double[weights.Length];
            for (int c = 0; c < weights.Length; c++)
            {
                double score = biases[c];
                for (int d = 0; d < weights[c].Length; d++)
                    score += weights[c][d] * ((sample[d] - mean[d]) / std[d]);
                raw[c] = score;
            }
            double max = raw.Max(), sum = 0;
            for (int c = 0; c < raw.Length; c++) { exp[c] = Math.Exp(raw[c] - max); sum += exp[c]; }
            for (int c = 0; c < exp.Length; c++) exp[c] /= sum;
            return exp;
        });
    }

    private void EnsureTrained()
    {
        if (_weights.Length == 0) throw new InvalidOperationException("Model henüz eğitilmedi.");
    }
}
