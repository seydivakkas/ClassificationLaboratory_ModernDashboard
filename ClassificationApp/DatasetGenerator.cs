namespace ClassificationApp;

public static class DatasetGenerator
{
    public static List<SamplePoint> LinearlySeparable(int classCount, int perClass, double noise, int seed)
    {
        var rnd = new Random(seed);
        var result = new List<SamplePoint>(classCount * perClass);
        double radius = classCount <= 2 ? 2.5 : 3.0;
        for (int c = 0; c < classCount; c++)
        {
            double angle = classCount == 2 ? (c == 0 ? Math.PI * 1.15 : Math.PI * 0.15) : 2 * Math.PI * c / classCount;
            double cx = Math.Cos(angle) * radius;
            double cy = Math.Sin(angle) * radius;
            for (int i = 0; i < perClass; i++)
            {
                double x = cx + Gaussian(rnd) * noise;
                double y = cy + Gaussian(rnd) * noise;
                result.Add(new SamplePoint(Clamp(x), Clamp(y), c));
            }
        }
        return result;
    }

    public static List<SamplePoint> Xor(int perClass, double noise, int seed)
    {
        var rnd = new Random(seed);
        var result = new List<SamplePoint>(perClass * 2);
        var centers = new[] { (-2.2, -2.2, 0), (2.2, 2.2, 0), (-2.2, 2.2, 1), (2.2, -2.2, 1) };
        int each = Math.Max(2, perClass / 2);
        foreach (var (cx, cy, cls) in centers)
            for (int i = 0; i < each; i++)
                result.Add(new SamplePoint(Clamp(cx + Gaussian(rnd) * noise), Clamp(cy + Gaussian(rnd) * noise), cls));
        return result;
    }

    public static List<SamplePoint> Clusters(int classCount, int perClass, double noise, int seed)
    {
        var rnd = new Random(seed);
        var result = new List<SamplePoint>(classCount * perClass);
        var centers = new (double X, double Y)[] { (-2.7, -1.9), (2.5, 1.9), (-1.7, 2.8), (2.8, -2.4), (0.0, 0.0) };
        for (int c = 0; c < classCount; c++)
            for (int i = 0; i < perClass; i++)
                result.Add(new SamplePoint(
                    Clamp(centers[c].X + Gaussian(rnd) * noise),
                    Clamp(centers[c].Y + Gaussian(rnd) * noise), c));
        return result;
    }

    public static List<SamplePoint> Spiral(int classCount, int perClass, double noise, int seed)
    {
        var rnd = new Random(seed);
        var result = new List<SamplePoint>(classCount * perClass);
        for (int c = 0; c < classCount; c++)
        {
            for (int i = 0; i < perClass; i++)
            {
                double t = perClass <= 1 ? 0 : (double)i / (perClass - 1);
                double radius = 0.35 + 4.0 * t;
                double angle = c * 2.0 * Math.PI / classCount + t * 2.7 * Math.PI;
                double x = radius * Math.Cos(angle) + Gaussian(rnd) * noise * 0.35;
                double y = radius * Math.Sin(angle) + Gaussian(rnd) * noise * 0.35;
                result.Add(new SamplePoint(Clamp(x), Clamp(y), c));
            }
        }
        return result;
    }

    private static double Clamp(double v) => Math.Clamp(v, -4.75, 4.75);

    private static double Gaussian(Random rnd)
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
