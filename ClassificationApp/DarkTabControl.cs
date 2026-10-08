namespace ClassificationApp;

public sealed class DarkTabControl : TabControl
{
    public DarkTabControl()
    {
        DrawMode = TabDrawMode.OwnerDrawFixed;
        ItemSize = new Size(88, 30);
        SizeMode = TabSizeMode.Fixed;
        BackColor = UiTheme.Surface;
        ForeColor = UiTheme.Text;
        DrawItem += DrawDarkItem;
    }

    protected override void OnPaintBackground(PaintEventArgs pevent)
    {
        pevent.Graphics.Clear(UiTheme.Surface);
    }

    private void DrawDarkItem(object? sender, DrawItemEventArgs e)
    {
        var page = TabPages[e.Index];
        bool selected = e.Index == SelectedIndex;
        using var fill = new SolidBrush(selected ? UiTheme.AccentSoft : UiTheme.Surface2);
        using var text = new SolidBrush(selected ? UiTheme.Text : UiTheme.Muted);
        e.Graphics.FillRectangle(fill, e.Bounds);
        using var font = new Font("Segoe UI Semibold", 8.5f);
        var size = e.Graphics.MeasureString(page.Text, font);
        e.Graphics.DrawString(page.Text, font, text,
            e.Bounds.Left + (e.Bounds.Width - size.Width) / 2,
            e.Bounds.Top + (e.Bounds.Height - size.Height) / 2);
    }
}
