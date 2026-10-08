using ML.Core.Classification;

namespace ClassificationApp;

public sealed class PlotPanel : Panel
{
    private IClassifier? _classifier;
    private Bitmap? _decisionCache;
    private SamplePoint? _dragging;
    private Point _lastMouse;
    private HashSet<SamplePoint> _misclassified = [];

    public List<SamplePoint> Samples { get; } = [];
    public List<ClassVisualStyle> ClassStyles { get; } =
    [
        new(Color.FromArgb(255, 92, 112), MarkerShape.Circle),
        new(Color.FromArgb(51, 153, 255), MarkerShape.Square),
        new(Color.FromArgb(50, 205, 145), MarkerShape.Triangle),
        new(Color.FromArgb(255, 190, 75), MarkerShape.Diamond),
        new(Color.FromArgb(180, 120, 255), MarkerShape.Cross)
    ];

    public IClassifier? Classifier
    {
        get => _classifier;
        set
        {
            if (ReferenceEquals(_classifier, value)) return;
            _classifier = value;
            InvalidateDecisionCache();
            Invalidate();
        }
    }

    public int SelectedClass { get; set; }
    public int ClassCount { get; set; } = 3;
    public bool ShowDecisionRegions { get; set; } = true;
    public bool ShowBoundaryLines { get; set; } = true;
    public bool ShowMisclassified { get; set; } = true;
    public bool ShowSplitStyle { get; set; } = true;
    public bool ModelStale { get; set; }

    public event EventHandler? SamplesChanged;
    public event Action<double, double>? CursorWorldChanged;

    public PlotPanel()
    {
        DoubleBuffered = true;
        BackColor = UiTheme.Plot;
        Cursor = Cursors.Cross;
        Resize += (_, _) => InvalidateDecisionCache();
        MouseDown += HandleMouseDown;
        MouseMove += HandleMouseMove;
        MouseUp += (_, _) => _dragging = null;
        MouseLeave += (_, _) => { _dragging = null; };
    }

    public void SetMisclassified(IEnumerable<SamplePoint> samples)
    {
        _misclassified = samples.ToHashSet();
        Invalidate();
    }

    public void ClearEvaluation()
    {
        _misclassified.Clear();
        Invalidate();
    }

    public void RefreshDecisionVisuals()
    {
        InvalidateDecisionCache();
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        e.Graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        DrawDecisionRegions(e.Graphics);
        DrawGrid(e.Graphics);
        DrawSamples(e.Graphics);
        DrawOverlay(e.Graphics);
    }

    private void DrawDecisionRegions(Graphics g)
    {
        if (!ShowDecisionRegions || _classifier is null || ClientSize.Width <= 0 || ClientSize.Height <= 0)
            return;

        if (_decisionCache is null || _decisionCache.Size != ClientSize)
            _decisionCache = BuildDecisionCache();
        if (_decisionCache is not null)
            g.DrawImageUnscaled(_decisionCache, 0, 0);
    }

    private Bitmap BuildDecisionCache()
    {
        var bmp = new Bitmap(Math.Max(1, Width), Math.Max(1, Height));
        if (_classifier is null) return bmp;
        const int step = 10;
        int cols = (Width + step - 1) / step;
        int rows = (Height + step - 1) / step;
        var classes = new int[cols, rows];

        using var gg = Graphics.FromImage(bmp);
        gg.Clear(UiTheme.Plot);
        gg.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;

        for (int cy = 0; cy < rows; cy++)
        for (int cx = 0; cx < cols; cx++)
        {
            int px = Math.Min(Width - 1, cx * step + step / 2);
            int py = Math.Min(Height - 1, cy * step + step / 2);
            var (x, y) = PixelToWorld(px, py);
            int cls;
            try { cls = Math.Clamp(_classifier.Predict([x, y]), 0, Math.Min(ClassCount, ClassStyles.Count) - 1); }
            catch { cls = 0; }
            classes[cx, cy] = cls;
            using var brush = new SolidBrush(Color.FromArgb(48, ClassStyles[cls].Color));
            gg.FillRectangle(brush, cx * step, cy * step, step + 1, step + 1);
        }

        if (ShowBoundaryLines)
        {
            using var boundary = new Pen(Color.FromArgb(145, 210, 230, 255), 1.2f);
            for (int cy = 0; cy < rows; cy++)
            for (int cx = 0; cx < cols; cx++)
            {
                int x = cx * step, y = cy * step;
                if (cx + 1 < cols && classes[cx, cy] != classes[cx + 1, cy])
                    gg.DrawLine(boundary, x + step, y, x + step, Math.Min(Height, y + step));
                if (cy + 1 < rows && classes[cx, cy] != classes[cx, cy + 1])
                    gg.DrawLine(boundary, x, y + step, Math.Min(Width, x + step), y + step);
            }
        }
        return bmp;
    }

    private void DrawGrid(Graphics g)
    {
        using var gridPen = new Pen(Color.FromArgb(35, 93, 132, 180));
        using var axisPen = new Pen(Color.FromArgb(135, 164, 202), 1.35f);
        using var labelBrush = new SolidBrush(Color.FromArgb(115, 148, 188));
        using var font = new Font("Segoe UI", 7.5f);

        for (int i = 1; i < 10; i++)
        {
            int x = i * Width / 10;
            int y = i * Height / 10;
            g.DrawLine(gridPen, x, 0, x, Height);
            g.DrawLine(gridPen, 0, y, Width, y);
        }

        int x0 = Width / 2, y0 = Height / 2;
        g.DrawLine(axisPen, x0, 0, x0, Height);
        g.DrawLine(axisPen, 0, y0, Width, y0);

        for (int v = -4; v <= 4; v += 2)
        {
            var xp = WorldToPixel(v, 0);
            var yp = WorldToPixel(0, v);
            if (v != 0)
            {
                g.DrawString(v.ToString(), font, labelBrush, xp.X + 2, y0 + 3);
                g.DrawString(v.ToString(), font, labelBrush, x0 + 3, yp.Y + 1);
            }
        }
    }

    private void DrawSamples(Graphics g)
    {
        foreach (var sample in Samples)
        {
            var p = WorldToPixel(sample.X, sample.Y);
            int cls = Math.Clamp(sample.ClassIndex, 0, ClassStyles.Count - 1);
            var style = ClassStyles[cls];

            if (ShowMisclassified && _misclassified.Contains(sample))
            {
                using var halo = new Pen(UiTheme.Danger, 3.5f);
                g.DrawEllipse(halo, p.X - 10, p.Y - 10, 20, 20);
            }

            bool test = ShowSplitStyle && sample.IsTest;
            DrawMarker(g, p, style, test, 12);
        }

        if (_dragging is not null)
        {
            var p = WorldToPixel(_dragging.X, _dragging.Y);
            using var pen = new Pen(UiTheme.Warning, 2f);
            g.DrawEllipse(pen, p.X - 9, p.Y - 9, 18, 18);
        }
    }

    private static void DrawMarker(Graphics g, PointF p, ClassVisualStyle style, bool test, int size)
    {
        float h = size / 2f;
        using var fill = new SolidBrush(test ? UiTheme.Plot : style.Color);
        using var outline = new Pen(style.Color, test ? 2.5f : 1.5f);
        using var inner = new Pen(Color.FromArgb(230, 245, 255), 1f);

        switch (style.Marker)
        {
            case MarkerShape.Circle:
                g.FillEllipse(fill, p.X - h, p.Y - h, size, size);
                g.DrawEllipse(outline, p.X - h, p.Y - h, size, size);
                break;
            case MarkerShape.Square:
                g.FillRectangle(fill, p.X - h, p.Y - h, size, size);
                g.DrawRectangle(outline, p.X - h, p.Y - h, size, size);
                break;
            case MarkerShape.Triangle:
                var tri = new[] { new PointF(p.X, p.Y - h - 1), new PointF(p.X + h + 1, p.Y + h), new PointF(p.X - h - 1, p.Y + h) };
                g.FillPolygon(fill, tri); g.DrawPolygon(outline, tri);
                break;
            case MarkerShape.Diamond:
                var dia = new[] { new PointF(p.X, p.Y - h - 1), new PointF(p.X + h + 1, p.Y), new PointF(p.X, p.Y + h + 1), new PointF(p.X - h - 1, p.Y) };
                g.FillPolygon(fill, dia); g.DrawPolygon(outline, dia);
                break;
            case MarkerShape.Cross:
                g.DrawLine(outline, p.X - h, p.Y - h, p.X + h, p.Y + h);
                g.DrawLine(outline, p.X + h, p.Y - h, p.X - h, p.Y + h);
                break;
        }

        if (test)
            g.DrawEllipse(inner, p.X - 2, p.Y - 2, 4, 4);
    }

    private void DrawOverlay(Graphics g)
    {
        using var font = new Font("Segoe UI Semibold", 8f);
        using var muted = new SolidBrush(UiTheme.Muted);
        g.DrawString("SOL TIK: EKLE / SÜRÜKLE    SAĞ TIK: SİL", font, muted, 12, Height - 24);

        if (ModelStale && _classifier is not null)
        {
            const string text = "MODEL ESKİ • yeniden eğit";
            var size = g.MeasureString(text, font);
            var rect = new RectangleF(Width - size.Width - 30, 12, size.Width + 18, 25);
            using var b = new SolidBrush(Color.FromArgb(210, UiTheme.Warning));
            using var dark = new SolidBrush(Color.FromArgb(40, 38, 18));
            g.FillRectangle(b, rect);
            g.DrawString(text, font, dark, rect.X + 9, rect.Y + 5);
        }
    }

    private void HandleMouseDown(object? sender, MouseEventArgs e)
    {
        _lastMouse = e.Location;
        if (e.Button == MouseButtons.Right)
        {
            RemoveNearest(e.Location);
            return;
        }
        if (e.Button != MouseButtons.Left) return;

        _dragging = FindNearest(e.Location, 15);
        if (_dragging is null)
        {
            var (x, y) = PixelToWorld(e.X, e.Y);
            var sample = new SamplePoint(x, y, Math.Clamp(SelectedClass, 0, ClassCount - 1));
            Samples.Add(sample);
            _dragging = sample;
            SamplesChanged?.Invoke(this, EventArgs.Empty);
        }
        Invalidate();
    }

    private void HandleMouseMove(object? sender, MouseEventArgs e)
    {
        _lastMouse = e.Location;
        var world = PixelToWorld(e.X, e.Y);
        CursorWorldChanged?.Invoke(world.X, world.Y);
        if (_dragging is null || e.Button != MouseButtons.Left) return;
        _dragging.X = Math.Clamp(world.X, -5.0, 5.0);
        _dragging.Y = Math.Clamp(world.Y, -5.0, 5.0);
        SamplesChanged?.Invoke(this, EventArgs.Empty);
        Invalidate();
    }

    private SamplePoint? FindNearest(Point location, double radius)
    {
        SamplePoint? best = null;
        double bestDist = radius * radius;
        foreach (var sample in Samples)
        {
            var p = WorldToPixel(sample.X, sample.Y);
            double dx = p.X - location.X, dy = p.Y - location.Y;
            double d = dx * dx + dy * dy;
            if (d < bestDist) { bestDist = d; best = sample; }
        }
        return best;
    }

    private void RemoveNearest(Point location)
    {
        var sample = FindNearest(location, 20);
        if (sample is null) return;
        Samples.Remove(sample);
        _misclassified.Remove(sample);
        SamplesChanged?.Invoke(this, EventArgs.Empty);
        Invalidate();
    }

    public (double X, double Y) PixelToWorld(int px, int py)
    {
        double x = (px / Math.Max(1.0, Width)) * 10.0 - 5.0;
        double y = 5.0 - (py / Math.Max(1.0, Height)) * 10.0;
        return (x, y);
    }

    public PointF WorldToPixel(double x, double y)
    {
        float px = (float)((x + 5.0) / 10.0 * Width);
        float py = (float)((5.0 - y) / 10.0 * Height);
        return new PointF(px, py);
    }

    public void ClearAll()
    {
        Samples.Clear();
        _misclassified.Clear();
        Classifier = null;
        ModelStale = false;
        SamplesChanged?.Invoke(this, EventArgs.Empty);
        Invalidate();
    }

    private void InvalidateDecisionCache()
    {
        _decisionCache?.Dispose();
        _decisionCache = null;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _decisionCache?.Dispose();
        base.Dispose(disposing);
    }
}
