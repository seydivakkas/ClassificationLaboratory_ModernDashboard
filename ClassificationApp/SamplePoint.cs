namespace ClassificationApp;

public sealed class SamplePoint
{
    public double X { get; set; }
    public double Y { get; set; }
    public int ClassIndex { get; set; }
    public bool IsTest { get; set; }

    public SamplePoint(double x, double y, int classIndex)
    {
        X = x;
        Y = y;
        ClassIndex = classIndex;
    }
}
