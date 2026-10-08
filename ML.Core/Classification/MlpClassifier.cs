using ML.Core.Models;
using ML.Core.Preprocessing;

namespace ML.Core.Classification;

public sealed class MlpClassifier : IProgressiveClassifier
{
    private readonly Random _random;
    private readonly ZScoreScaler _scaler = new();
    private double[,] _w1 = new double[0, 0];
    private double[] _b1 = Array.Empty<double>();
    private double[,] _w2 = new double[0, 0];
    private double[] _b2 = Array.Empty<double>();

    public int HiddenNeurons { get; }
    public double LearningRate { get; }
    public int MaxEpochs { get; }
    public double Momentum { get; }
    public int ClassCount => _b2.Length;

    public MlpClassifier(int hiddenNeurons = 8, double learningRate = 0.001, int maxEpochs = 10_000, double momentum = 0.9, int seed = 42)
    {
        if (hiddenNeurons <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenNeurons));
        if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (maxEpochs <= 0) throw new ArgumentOutOfRangeException(nameof(maxEpochs));
        if (momentum is < 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(momentum));
        HiddenNeurons = hiddenNeurons;
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
        int inputDim = ClassifierValidation.Validate(samples, labels);
        int outputDim = labels.Max() + 1;
        snapshotEvery = Math.Max(1, snapshotEvery);
        _scaler.Fit(samples);
        var x = _scaler.Transform(samples);
        InitializeWeights(inputDim, outputDim);

        var vw1 = new double[inputDim, HiddenNeurons];
        var vb1 = new double[HiddenNeurons];
        var vw2 = new double[HiddenNeurons, outputDim];
        var vb2 = new double[outputDim];
        var history = new List<double>(MaxEpochs);

        int epoch;
        for (epoch = 1; epoch <= MaxEpochs; epoch++)
        {
            double totalLoss = 0.0;
            for (int n = 0; n < x.Length; n++)
            {
                var hidden = new double[HiddenNeurons];
                for (int h = 0; h < HiddenNeurons; h++)
                {
                    double z = _b1[h];
                    for (int i = 0; i < inputDim; i++) z += x[n][i] * _w1[i, h];
                    hidden[h] = Sigmoid(z);
                }

                var logits = new double[outputDim];
                for (int o = 0; o < outputDim; o++)
                {
                    double z = _b2[o];
                    for (int h = 0; h < HiddenNeurons; h++) z += hidden[h] * _w2[h, o];
                    logits[o] = z;
                }

                var probs = Softmax(logits);
                int target = labels[n];
                totalLoss += -Math.Log(probs[target] + 1e-12);

                var deltaOut = new double[outputDim];
                for (int o = 0; o < outputDim; o++) deltaOut[o] = probs[o] - (o == target ? 1.0 : 0.0);
                var deltaHidden = new double[HiddenNeurons];
                for (int h = 0; h < HiddenNeurons; h++)
                {
                    double error = 0.0;
                    for (int o = 0; o < outputDim; o++) error += deltaOut[o] * _w2[h, o];
                    deltaHidden[h] = error * hidden[h] * (1.0 - hidden[h]);
                }

                for (int h = 0; h < HiddenNeurons; h++)
                    for (int o = 0; o < outputDim; o++)
                    {
                        double update = -LearningRate * (deltaOut[o] * hidden[h]) + Momentum * vw2[h, o];
                        _w2[h, o] += update;
                        vw2[h, o] = update;
                    }

                for (int o = 0; o < outputDim; o++)
                {
                    double update = -LearningRate * deltaOut[o] + Momentum * vb2[o];
                    _b2[o] += update;
                    vb2[o] = update;
                }

                for (int i = 0; i < inputDim; i++)
                    for (int h = 0; h < HiddenNeurons; h++)
                    {
                        double update = -LearningRate * (deltaHidden[h] * x[n][i]) + Momentum * vw1[i, h];
                        _w1[i, h] += update;
                        vw1[i, h] = update;
                    }

                for (int h = 0; h < HiddenNeurons; h++)
                {
                    double update = -LearningRate * deltaHidden[h] + Momentum * vb1[h];
                    _b1[h] += update;
                    vb1[h] = update;
                }
            }

            double avgLoss = totalLoss / x.Length;
            history.Add(avgLoss);
            if (onSnapshot is not null && (epoch <= 12 || epoch % snapshotEvery == 0 || avgLoss < 1e-4))
                onSnapshot(new ClassifierEpochSnapshot { Epoch = epoch, Loss = avgLoss, Model = CreateSnapshot() });
            if (avgLoss < 1e-4) break;
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
        var probs = PredictScores(sample);
        int best = 0;
        for (int i = 1; i < probs.Length; i++) if (probs[i] > probs[best]) best = i;
        return best;
    }

    public double[] PredictScores(IReadOnlyList<double> sample)
    {
        EnsureTrained();
        var x = _scaler.Transform(sample);
        return PredictNormalizedScores(x, _w1, _b1, _w2, _b2);
    }

    private void InitializeWeights(int inputDim, int outputDim)
    {
        _w1 = new double[inputDim, HiddenNeurons];
        _b1 = new double[HiddenNeurons];
        _w2 = new double[HiddenNeurons, outputDim];
        _b2 = new double[outputDim];
        double limit1 = Math.Sqrt(6.0 / (inputDim + HiddenNeurons));
        double limit2 = Math.Sqrt(6.0 / (HiddenNeurons + outputDim));
        for (int i = 0; i < inputDim; i++) for (int h = 0; h < HiddenNeurons; h++) _w1[i, h] = Uniform(-limit1, limit1);
        for (int h = 0; h < HiddenNeurons; h++) for (int o = 0; o < outputDim; o++) _w2[h, o] = Uniform(-limit2, limit2);
    }

    private IClassifier CreateSnapshot()
    {
        var w1 = CloneMatrix(_w1); var b1 = _b1.ToArray(); var w2 = CloneMatrix(_w2); var b2 = _b2.ToArray();
        var mean = _scaler.Mean.ToArray(); var std = _scaler.Std.ToArray();
        return new FrozenClassifier(sample =>
        {
            var x = new double[mean.Length];
            for (int i = 0; i < x.Length; i++) x[i] = (sample[i] - mean[i]) / std[i];
            return PredictNormalizedScores(x, w1, b1, w2, b2);
        });
    }

    private static double[] PredictNormalizedScores(double[] x, double[,] w1, double[] b1, double[,] w2, double[] b2)
    {
        int hiddenCount = b1.Length;
        var hidden = new double[hiddenCount];
        for (int h = 0; h < hiddenCount; h++)
        {
            double z = b1[h];
            for (int i = 0; i < x.Length; i++) z += x[i] * w1[i, h];
            hidden[h] = Sigmoid(z);
        }
        var logits = new double[b2.Length];
        for (int o = 0; o < b2.Length; o++)
        {
            double z = b2[o];
            for (int h = 0; h < hiddenCount; h++) z += hidden[h] * w2[h, o];
            logits[o] = z;
        }
        return Softmax(logits);
    }

    private static double[,] CloneMatrix(double[,] source)
    {
        int r = source.GetLength(0), c = source.GetLength(1);
        var clone = new double[r, c];
        Array.Copy(source, clone, source.Length);
        return clone;
    }

    private double Uniform(double min, double max) => min + _random.NextDouble() * (max - min);
    private static double Sigmoid(double x) { x = Math.Clamp(x, -40.0, 40.0); return 1.0 / (1.0 + Math.Exp(-x)); }
    private static double[] Softmax(IReadOnlyList<double> logits)
    {
        double max = logits.Max(); var exp = new double[logits.Count]; double sum = 0.0;
        for (int i = 0; i < logits.Count; i++) { exp[i] = Math.Exp(logits[i] - max); sum += exp[i]; }
        for (int i = 0; i < exp.Length; i++) exp[i] /= sum;
        return exp;
    }
    private void EnsureTrained() { if (_w1.Length == 0) throw new InvalidOperationException("Model henüz eğitilmedi."); }
}
