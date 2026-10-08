namespace ClassificationApp;

public sealed class LossChartPanel : Panel
{
    private IReadOnlyList<double> _values = Array.Empty<double>();
    private int _displayedCount;

    public IReadOnlyList<double> Values
    {
        get => _values;
        set { _values = value ?? Array.Empty<double>(); _displayedCount = 0; Invalidate(); }
    }

    public int DisplayedCount
    {
        get => _displayedCount;
        set { _displayedCount = value; Invalidate(); }
    }

    public LossChartPanel()
    {
        DoubleBuffered = true;
        BackColor = UiTheme.Surface;
        Padding = new Padding(8);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var g = e.Graphics;
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        using var title = new Font("Segoe UI Semibold", 9.5f);
        using var small = new Font("Segoe UI", 7.5f);
        using var text = new SolidBrush(UiTheme.Text);
        using var muted = new SolidBrush(UiTheme.Muted);
        g.DrawString("Training loss / error", title, text, 10, 8);

        int count = _displayedCount > 0 ? Math.Min(_displayedCount, _values.Count) : _values.Count;
        var values = _values.Take(count).Where(double.IsFinite).ToArray();
        if (values.Length < 2)
        {
            g.DrawString("Henüz eğitim verisi yok.", small, muted, 10, 34);
            return;
        }

        int left = 44, top = 32, right = 12, bottom = 25;
        int w = Math.Max(1, Width - left - right);
        int h = Math.Max(1, Height - top - bottom);
        double min = Math.Min(0, values.Min());
        double max = values.Max();
        if (Math.Abs(max - min) < 1e-12) max = min + 1;

        using var grid = new Pen(Color.FromArgb(35, UiTheme.Muted));
        using var axis = new Pen(Color.FromArgb(90, UiTheme.Muted));
        for (int i = 0; i <= 4; i++)
        {
            int y = top + i * h / 4;
            g.DrawLine(grid, left, y, left + w, y);
        }
        g.DrawLine(axis, left, top, left, top + h);
        g.DrawLine(axis, left, top + h, left + w, top + h);

        var pts = new PointF[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            float x = left + (float)i / (values.Length - 1) * w;
            float y = top + h - (float)((values[i] - min) / (max - min) * h);
            pts[i] = new PointF(x, y);
        }

        using var area = new System.Drawing.Drawing2D.GraphicsPath();
        area.AddLines(pts);
        area.AddLine(pts[^1].X, pts[^1].Y, pts[^1].X, top + h);
        area.AddLine(pts[^1].X, top + h, pts[0].X, top + h);
        area.CloseFigure();
        using var areaBrush = new SolidBrush(Color.FromArgb(35, UiTheme.Cyan));
        g.FillPath(areaBrush, area);

        using var line = new Pen(UiTheme.Cyan, 2.2f);
        g.DrawLines(line, pts);
        using var dot = new SolidBrush(UiTheme.Cyan);
        g.FillEllipse(dot, pts[^1].X - 3, pts[^1].Y - 3, 6, 6);

        g.DrawString(max.ToString("0.###"), small, muted, 2, top - 4);
        g.DrawString(min.ToString("0.###"), small, muted, 2, top + h - 10);
        g.DrawString($"1", small, muted, left, top + h + 5);
        g.DrawString($"{values.Length}", small, muted, Math.Max(left, left + w - 22), top + h + 5);
    }
}
