using ML.Core.Classification;
using ML.Core.Models;

namespace ClassificationApp;

public sealed class MainForm : Form
{
    private readonly PlotPanel _plot = new() { Dock = DockStyle.Fill };
    private readonly LossChartPanel _lossChart = new() { Dock = DockStyle.Fill };
    private readonly ConfusionMatrixPanel _confusion = new() { Dock = DockStyle.Fill };

    private readonly NumericUpDown _classCount = new() { Minimum = 2, Maximum = 5, Value = 3, Width = 92 };
    private readonly NumericUpDown _samplesPerClass = new() { Minimum = 4, Maximum = 250, Value = 35, Width = 92 };
    private readonly NumericUpDown _noise = new() { DecimalPlaces = 2, Increment = 0.05m, Minimum = 0.05m, Maximum = 2.5m, Value = 0.65m, Width = 92 };
    private readonly NumericUpDown _seed = new() { Minimum = 1, Maximum = 999999, Value = 42, Width = 92 };
    private readonly NumericUpDown _testPercent = new() { Minimum = 10, Maximum = 50, Value = 25, Increment = 5, Width = 92 };

    private readonly NumericUpDown _learningRate = new() { DecimalPlaces = 4, Increment = 0.0005m, Minimum = 0.0001m, Maximum = 1m, Value = 0.001m, Width = 108 };
    private readonly NumericUpDown _epochs = new() { Minimum = 1, Maximum = 100000, Value = 10000, Increment = 100, Width = 108 };
    private readonly NumericUpDown _momentum = new() { DecimalPlaces = 2, Increment = 0.05m, Minimum = 0m, Maximum = 0.99m, Value = 0.90m, Width = 108 };
    private readonly NumericUpDown _hidden = new() { Minimum = 2, Maximum = 128, Value = 12, Width = 108 };

    private readonly FlowLayoutPanel _classRows = new() { AutoSize = true, FlowDirection = FlowDirection.TopDown, WrapContents = false };
    private readonly Button[] _algorithmButtons = new Button[3];
    private readonly CheckBox _animateBoundary = Toggle("Epoch sınır animasyonu", true);
    private readonly CheckBox _showRegions = Toggle("Karar bölgelerini göster", true);
    private readonly CheckBox _showBoundary = Toggle("Sınır çizgilerini göster", true);
    private readonly CheckBox _showMisclassified = Toggle("Hatalı örnekleri vurgula", true);
    private readonly CheckBox _showSplit = Toggle("Train / test stilini göster", true);

    private readonly MetricCard _accuracyCard = new("Accuracy");
    private readonly MetricCard _f1Card = new("Macro F1");
    private readonly MetricCard _precisionCard = new("Precision");
    private readonly MetricCard _recallCard = new("Recall");
    private readonly MetricCard _epochCard = new("Epoch");
    private readonly MetricCard _lossCard = new("Loss / Error");

    private readonly Label _datasetName = UiTheme.Label("DATASET • Clusters", 8.5f, true, UiTheme.Cyan);
    private readonly Label _sampleSummary = UiTheme.Label("0 örnek", 8.5f, false, UiTheme.Muted);
    private readonly Label _splitSummary = UiTheme.Label("Train: -   Test: -", 8.5f, false, UiTheme.Muted);
    private readonly Label _cursorLabel = UiTheme.Label("x: 0.00   y: 0.00", 8.5f, false, UiTheme.Muted);
    private readonly RichTextBox _log = new() { ReadOnly = true, Dock = DockStyle.Fill, BorderStyle = BorderStyle.None };
    private readonly DataGridView _comparisonGrid = new() { Dock = DockStyle.Fill };
    private readonly ToolStripStatusLabel _statusText = new("Hazır");
    private readonly ToolStripProgressBar _statusProgress = new() { Style = ProgressBarStyle.Marquee, Visible = false, Width = 120 };

    private readonly Button _trainButton = UiTheme.Button("MODELİ EĞİT", true);
    private readonly Button _retrainButton = UiTheme.Button("AYNI VERİDE TEKRAR EĞİT");
    private readonly Button _resetModelButton = UiTheme.Button("Modeli sıfırla");
    private readonly Button _clearDataButton = UiTheme.Button("Veriyi temizle");
    private readonly Button _activateRunButton = UiTheme.Button("Seçili modeli göster", true);
    private readonly Button _clearRunsButton = UiTheme.Button("Karşılaştırmayı temizle");

    private readonly List<ModelRun> _runs = [];
    private readonly List<Control> _busySensitiveControls = [];
    private int _selectedAlgorithm = 1;
    private int _datasetVersion = 1;
    private int _runCounter;
    private bool _suppressDataDirty;
    private string _currentDataset = "Clusters";

    public MainForm()
    {
        Text = "Decision Boundary Explorer • Classification Laboratory";
        StartPosition = FormStartPosition.CenterScreen;
        MinimumSize = new Size(1280, 760);
        Size = new Size(1560, 930);
        Font = new Font("Segoe UI", 9.5f);
        BackColor = UiTheme.Window;
        ForeColor = UiTheme.Text;

        StyleInputs();
        ConfigureEvents();
        ConfigureComparisonGrid();
        ConfigureLog();

        var status = new StatusStrip { BackColor = UiTheme.Sidebar, ForeColor = UiTheme.Muted, SizingGrip = false };
        status.Items.Add(_statusText);
        status.Items.Add(new ToolStripStatusLabel { Spring = true });
        status.Items.Add(_statusProgress);
        status.Items.Add(new ToolStripStatusLabel("From-scratch C# • Z-Score • Momentum • Backpropagation"));

        var shell = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1, BackColor = UiTheme.Window };
        shell.RowStyles.Add(new RowStyle(SizeType.Absolute, 72));
        shell.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        shell.Controls.Add(BuildHeader(), 0, 0);
        shell.Controls.Add(BuildMainArea(), 0, 1);

        Controls.Add(shell);
        Controls.Add(status);
        status.Dock = DockStyle.Bottom;
        shell.Dock = DockStyle.Fill;
        status.BringToFront();

        RebuildClassRows();
        SelectAlgorithm(1, resetModel: false);
        GenerateDataset("Clusters");
    }

    private void StyleInputs()
    {
        foreach (var n in new[] { _classCount, _samplesPerClass, _noise, _seed, _testPercent, _learningRate, _epochs, _momentum, _hidden })
            UiTheme.StyleNumeric(n);

        _log.BackColor = UiTheme.Plot;
        _log.ForeColor = UiTheme.Muted;
        _log.Font = new Font("Cascadia Mono", 8.5f);
    }

    private void ConfigureEvents()
    {
        _classCount.ValueChanged += (_, _) => RebuildClassRows();
        _plot.SamplesChanged += (_, _) => OnSamplesChanged();
        _plot.CursorWorldChanged += (x, y) => _cursorLabel.Text = $"x: {x,6:0.00}   y: {y,6:0.00}";

        _trainButton.Click += async (_, _) => await TrainAsync();
        _retrainButton.Click += async (_, _) => await TrainAsync();
        _resetModelButton.Click += (_, _) => ResetActiveModel(clearLoss: true);
        _clearDataButton.Click += (_, _) => ClearData();
        _activateRunButton.Click += (_, _) => ActivateSelectedRun();
        _clearRunsButton.Click += (_, _) => { _runs.Clear(); RefreshComparisonGrid(); Log("Model karşılaştırma geçmişi temizlendi."); };
        _comparisonGrid.CellDoubleClick += (_, _) => ActivateSelectedRun();

        _showRegions.CheckedChanged += (_, _) => { _plot.ShowDecisionRegions = _showRegions.Checked; _plot.RefreshDecisionVisuals(); };
        _showBoundary.CheckedChanged += (_, _) => { _plot.ShowBoundaryLines = _showBoundary.Checked; _plot.RefreshDecisionVisuals(); };
        _showMisclassified.CheckedChanged += (_, _) => { _plot.ShowMisclassified = _showMisclassified.Checked; _plot.Invalidate(); };
        _showSplit.CheckedChanged += (_, _) => { _plot.ShowSplitStyle = _showSplit.Checked; _plot.Invalidate(); };

        _busySensitiveControls.AddRange([_classCount, _samplesPerClass, _noise, _seed, _testPercent, _learningRate, _epochs, _momentum, _hidden,
            _trainButton, _retrainButton, _resetModelButton, _clearDataButton, _animateBoundary, _showRegions, _showBoundary, _showMisclassified, _showSplit]);
    }

    private Control BuildHeader()
    {
        var header = new Panel { Dock = DockStyle.Fill, BackColor = UiTheme.Sidebar, Padding = new Padding(22, 11, 22, 8) };
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1 };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 75));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));

        var titleBox = new Panel { Dock = DockStyle.Fill };
        var title = UiTheme.Label("DECISION BOUNDARY EXPLORER", 18f, true);
        title.Location = new Point(0, 0);
        var subtitle = UiTheme.Label("Classification Laboratory • Binary Perceptron / Multiclass Perceptron / MLP", 8.5f, false, UiTheme.Muted);
        subtitle.Location = new Point(2, 34);
        titleBox.Controls.AddRange([title, subtitle]);

        var right = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.RightToLeft, WrapContents = false, Padding = new Padding(0, 12, 0, 0) };
        right.Controls.Add(_cursorLabel);
        right.Controls.Add(UiTheme.Label("CURSOR", 8f, true, UiTheme.Cyan));
        layout.Controls.Add(titleBox, 0, 0);
        layout.Controls.Add(right, 1, 0);
        header.Controls.Add(layout);
        return header;
    }

    private Control BuildMainArea()
    {
        var main = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 1, Padding = new Padding(10), BackColor = UiTheme.Window };
        main.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 292));
        main.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        main.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 392));
        main.Controls.Add(BuildLeftSidebar(), 0, 0);
        main.Controls.Add(BuildCenter(), 1, 0);
        main.Controls.Add(BuildRightSidebar(), 2, 0);
        return main;
    }

    private Control BuildLeftSidebar()
    {
        var host = new Panel { Dock = DockStyle.Fill, BackColor = UiTheme.Sidebar, AutoScroll = true, Padding = new Padding(4) };
        var stack = new FlowLayoutPanel { Dock = DockStyle.Top, AutoSize = true, FlowDirection = FlowDirection.TopDown, WrapContents = false, Padding = new Padding(2) };

        stack.Controls.Add(BuildDatasetCard());
        stack.Controls.Add(BuildClassCard());
        stack.Controls.Add(BuildSplitCard());
        stack.Controls.Add(BuildDisplayCard());

        _clearDataButton.Width = 248;
        stack.Controls.Add(_clearDataButton);
        host.Controls.Add(stack);
        _busySensitiveControls.Add(stack);
        return host;
    }

    private Control BuildDatasetCard()
    {
        var content = new TableLayoutPanel { AutoSize = true, ColumnCount = 1, Width = 244 };
        content.Controls.Add(UiTheme.Label("Hazır veri kümeleri", 8.5f, true, UiTheme.Muted));

        var grid = new TableLayoutPanel { AutoSize = true, ColumnCount = 2, Width = 244, Margin = new Padding(0, 7, 0, 8) };
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        string[] names = ["Linearly Separable", "XOR", "Clusters", "Spiral"];
        for (int i = 0; i < names.Length; i++)
        {
            string name = names[i];
            var b = UiTheme.Button(name); b.Dock = DockStyle.Fill; b.Height = 35; b.Font = new Font("Segoe UI Semibold", 8.2f);
            b.Click += (_, _) => GenerateDataset(name);
            grid.Controls.Add(b, i % 2, i / 2);
            _busySensitiveControls.Add(b);
        }
        content.Controls.Add(grid);
        content.Controls.Add(SmallField("Örnek / sınıf", _samplesPerClass));
        content.Controls.Add(SmallField("Noise", _noise));
        content.Controls.Add(SmallField("Random seed", _seed));
        return Card("DATASET", content, 266);
    }

    private Control BuildClassCard()
    {
        var content = new TableLayoutPanel { AutoSize = true, ColumnCount = 1, Width = 244 };
        content.Controls.Add(SmallField("Sınıf sayısı", _classCount));
        content.Controls.Add(UiTheme.Label("Aktif sınıf  •  renk  •  marker", 8f, true, UiTheme.Muted));
        content.Controls.Add(_classRows);
        return Card("CLASSES", content, 266);
    }

    private Control BuildSplitCard()
    {
        var content = new TableLayoutPanel { AutoSize = true, ColumnCount = 1, Width = 244 };
        content.Controls.Add(SmallField("Test oranı (%)", _testPercent));
        _splitSummary.Margin = new Padding(0, 8, 0, 2);
        content.Controls.Add(_splitSummary);
        content.Controls.Add(UiTheme.Label("Split sınıf bazında (stratified) uygulanır.", 7.7f, false, UiTheme.Muted));
        return Card("TRAIN / TEST", content, 266);
    }

    private Control BuildDisplayCard()
    {
        var content = new FlowLayoutPanel { AutoSize = true, FlowDirection = FlowDirection.TopDown, WrapContents = false, Width = 244 };
        content.Controls.AddRange([_showRegions, _showBoundary, _showMisclassified, _showSplit, _animateBoundary]);
        return Card("VISUALIZATION", content, 266);
    }

    private Control BuildCenter()
    {
        var card = new RoundedCard { Dock = DockStyle.Fill, Padding = new Padding(1), BackColor = UiTheme.Surface, Margin = new Padding(8) };
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1 };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 43));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var toolbar = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, Padding = new Padding(12, 7, 12, 4), BackColor = UiTheme.Surface };
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 45));
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));
        toolbar.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
        toolbar.Controls.Add(_datasetName, 0, 0);
        toolbar.Controls.Add(_sampleSummary, 1, 0);
        var legend = UiTheme.Label("● Train    ○ Test    ◎ Hatalı", 8.3f, false, UiTheme.Muted);
        legend.TextAlign = ContentAlignment.MiddleRight; legend.Dock = DockStyle.Fill;
        toolbar.Controls.Add(legend, 2, 0);

        layout.Controls.Add(toolbar, 0, 0);
        layout.Controls.Add(_plot, 0, 1);
        card.Controls.Add(layout);
        return card;
    }

    private Control BuildRightSidebar()
    {
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 3, ColumnCount = 1, BackColor = UiTheme.Sidebar, Padding = new Padding(5) };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 273));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 190));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        layout.Controls.Add(BuildModelCard(), 0, 0);
        layout.Controls.Add(BuildMetricsGrid(), 0, 1);
        layout.Controls.Add(BuildAnalysisTabs(), 0, 2);
        return layout;
    }

    private Control BuildModelCard()
    {
        var content = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 5, ColumnCount = 1 };
        content.RowStyles.Add(new RowStyle(SizeType.Absolute, 45));
        content.RowStyles.Add(new RowStyle(SizeType.Absolute, 105));
        content.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        content.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        content.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var alg = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3 };
        string[] names = ["Binary", "Multiclass", "MLP"];
        for (int i = 0; i < 3; i++)
        {
            int index = i;
            var b = UiTheme.Button(names[i]); b.Dock = DockStyle.Fill; b.Height = 34;
            b.Click += (_, _) => SelectAlgorithm(index, resetModel: true);
            _algorithmButtons[i] = b;
            alg.Controls.Add(b, i, 0);
            _busySensitiveControls.Add(b);
        }
        content.Controls.Add(alg, 0, 0);

        var settings = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 4, RowCount = 2, Padding = new Padding(0, 6, 0, 2) };
        settings.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));
        settings.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));
        settings.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));
        settings.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 25));
        AddCompactSetting(settings, "Learning rate", _learningRate, 0);
        AddCompactSetting(settings, "Max epoch", _epochs, 1);
        AddCompactSetting(settings, "Momentum", _momentum, 2);
        AddCompactSetting(settings, "Hidden", _hidden, 3);
        content.Controls.Add(settings, 0, 1);

        _trainButton.Dock = DockStyle.Fill; _trainButton.Margin = new Padding(3);
        _retrainButton.Dock = DockStyle.Fill; _retrainButton.Margin = new Padding(3);
        content.Controls.Add(_trainButton, 0, 2);
        content.Controls.Add(_retrainButton, 0, 3);

        _resetModelButton.Dock = DockStyle.Fill; _resetModelButton.Margin = new Padding(3);
        content.Controls.Add(_resetModelButton, 0, 4);
        return Card("MODEL & TRAINING", content, 370, dockFill: true);
    }

    private Control BuildMetricsGrid()
    {
        var card = new RoundedCard { Dock = DockStyle.Fill, Margin = new Padding(6), Padding = new Padding(8), BackColor = UiTheme.Surface };
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 2 };
        for (int i = 0; i < 3; i++) layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.333f));
        for (int i = 0; i < 2; i++) layout.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        MetricCard[] cards = [_accuracyCard, _f1Card, _precisionCard, _recallCard, _epochCard, _lossCard];
        for (int i = 0; i < cards.Length; i++) { cards[i].Dock = DockStyle.Fill; layout.Controls.Add(cards[i], i % 3, i / 3); }
        card.Controls.Add(layout);
        return card;
    }

    private Control BuildAnalysisTabs()
    {
        var tabs = new DarkTabControl { Dock = DockStyle.Fill, Margin = new Padding(6) };
        tabs.TabPages.Add(CreateTab("LOSS", _lossChart));
        tabs.TabPages.Add(CreateTab("MATRIX", _confusion));
        tabs.TabPages.Add(CreateTab("COMPARE", BuildComparisonView()));
        tabs.TabPages.Add(CreateTab("LOG", _log));
        return tabs;
    }

    private Control BuildComparisonView()
    {
        var panel = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1, BackColor = UiTheme.Surface };
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        panel.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
        panel.Controls.Add(_comparisonGrid, 0, 0);
        var buttons = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 };
        _activateRunButton.Dock = DockStyle.Fill; _clearRunsButton.Dock = DockStyle.Fill;
        buttons.Controls.Add(_activateRunButton, 0, 0); buttons.Controls.Add(_clearRunsButton, 1, 0);
        panel.Controls.Add(buttons, 0, 1);
        return panel;
    }

    private static TabPage CreateTab(string title, Control content)
    {
        var page = new TabPage(title) { BackColor = UiTheme.Surface, ForeColor = UiTheme.Text, Padding = new Padding(4) };
        content.Dock = DockStyle.Fill;
        page.Controls.Add(content);
        return page;
    }

    private RoundedCard Card(string title, Control content, int width, bool dockFill = false)
    {
        var card = new RoundedCard { Width = width, AutoSize = !dockFill, AutoSizeMode = AutoSizeMode.GrowAndShrink, BackColor = UiTheme.Surface };
        if (dockFill) card.Dock = DockStyle.Fill;
        var layout = new TableLayoutPanel { AutoSize = !dockFill, Dock = dockFill ? DockStyle.Fill : DockStyle.Top, ColumnCount = 1, Width = width - 24 };
        if (dockFill) { layout.RowCount = 2; layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 26)); layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); }
        var label = UiTheme.Label(title, 8f, true, UiTheme.Cyan); label.Margin = new Padding(0, 0, 0, 7);
        layout.Controls.Add(label, 0, 0);
        layout.Controls.Add(content, 0, 1);
        card.Controls.Add(layout);
        return card;
    }

    private static Control SmallField(string label, Control input)
    {
        var row = new TableLayoutPanel { AutoSize = true, ColumnCount = 2, Width = 240, Margin = new Padding(0, 4, 0, 4) };
        row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 58));
        row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 42));
        var l = UiTheme.Label(label, 8.5f, false, UiTheme.Muted); l.Anchor = AnchorStyles.Left;
        input.Anchor = AnchorStyles.Right;
        row.Controls.Add(l, 0, 0); row.Controls.Add(input, 1, 0);
        return row;
    }

    private static void AddCompactSetting(TableLayoutPanel table, string caption, Control control, int column)
    {
        var box = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1, Margin = new Padding(3) };
        var l = UiTheme.Label(caption, 7.3f, true, UiTheme.Muted); l.Dock = DockStyle.Bottom;
        control.Dock = DockStyle.Top;
        box.Controls.Add(l, 0, 0); box.Controls.Add(control, 0, 1);
        table.Controls.Add(box, column, 0);
        table.SetRowSpan(box, 2);
    }

    private static CheckBox Toggle(string text, bool value)
        => new()
        {
            Text = text,
            Checked = value,
            AutoSize = true,
            ForeColor = UiTheme.Text,
            BackColor = Color.Transparent,
            Font = new Font("Segoe UI", 8.5f),
            Margin = new Padding(1, 4, 1, 4)
        };

    private void RebuildClassRows()
    {
        _classRows.Controls.Clear();
        int count = (int)_classCount.Value;
        _plot.ClassCount = count;
        if (_plot.SelectedClass >= count) _plot.SelectedClass = 0;

        bool removed = _plot.Samples.RemoveAll(p => p.ClassIndex >= count) > 0;
        if (removed && !_suppressDataDirty) MarkDatasetChanged("Sınıf sayısı değişti; kapsam dışı örnekler kaldırıldı.");

        for (int c = 0; c < count; c++)
        {
            int classIndex = c;
            var row = new FlowLayoutPanel { AutoSize = true, FlowDirection = FlowDirection.LeftToRight, WrapContents = false, Width = 242, Height = 32, Margin = new Padding(0, 3, 0, 3) };
            var radio = new RadioButton
            {
                Text = $"Class {c}", Width = 72, AutoSize = false, Checked = c == _plot.SelectedClass,
                ForeColor = UiTheme.Text, BackColor = Color.Transparent, Font = new Font("Segoe UI", 8.5f)
            };
            radio.CheckedChanged += (_, _) => { if (radio.Checked) _plot.SelectedClass = classIndex; };

            var color = UiTheme.Button("");
            color.Width = 31; color.Height = 26; color.Margin = new Padding(2, 0, 4, 0); color.BackColor = _plot.ClassStyles[c].Color;
            color.FlatAppearance.BorderColor = Color.FromArgb(180, 220, 240);
            color.Click += (_, _) => PickClassColor(classIndex, color);

            var marker = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 100, Height = 27 };
            marker.Items.AddRange(Enum.GetNames<MarkerShape>());
            marker.SelectedIndex = (int)_plot.ClassStyles[c].Marker;
            UiTheme.StyleCombo(marker);
            marker.SelectedIndexChanged += (_, _) =>
            {
                if (marker.SelectedIndex >= 0)
                {
                    _plot.ClassStyles[classIndex].Marker = (MarkerShape)marker.SelectedIndex;
                    _plot.Invalidate();
                }
            };
            row.Controls.AddRange([radio, color, marker]);
            _classRows.Controls.Add(row);
            _busySensitiveControls.Add(radio); _busySensitiveControls.Add(color); _busySensitiveControls.Add(marker);
        }
        _plot.RefreshDecisionVisuals();
        UpdateAlgorithmUi();
        UpdateSampleSummary();
    }

    private void PickClassColor(int classIndex, Button preview)
    {
        using var dialog = new ColorDialog { Color = _plot.ClassStyles[classIndex].Color, FullOpen = true };
        if (dialog.ShowDialog(this) != DialogResult.OK) return;
        _plot.ClassStyles[classIndex].Color = dialog.Color;
        preview.BackColor = dialog.Color;
        _plot.RefreshDecisionVisuals();
    }

    private void SelectAlgorithm(int index, bool resetModel)
    {
        _selectedAlgorithm = Math.Clamp(index, 0, 2);
        for (int i = 0; i < _algorithmButtons.Length; i++)
        {
            if (_algorithmButtons[i] is null) continue;
            bool selected = i == _selectedAlgorithm;
            _algorithmButtons[i].BackColor = selected ? UiTheme.Accent : UiTheme.Surface2;
            _algorithmButtons[i].FlatAppearance.BorderColor = selected ? UiTheme.Cyan : UiTheme.Border;
        }

        if (_selectedAlgorithm == 0 && _classCount.Value != 2)
            _classCount.Value = 2;
        UpdateAlgorithmUi();
        if (resetModel) ResetActiveModel(clearLoss: false);
        Log($"Algoritma seçildi: {CurrentAlgorithmName()}.");
    }

    private void UpdateAlgorithmUi()
    {
        _classCount.Enabled = _selectedAlgorithm != 0;
        _hidden.Enabled = _selectedAlgorithm == 2;
        _plot.ClassCount = _selectedAlgorithm == 0 ? 2 : (int)_classCount.Value;
    }

    private void GenerateDataset(string kind)
    {
        if (kind == "XOR" && _classCount.Value != 2)
        {
            _suppressDataDirty = true;
            _classCount.Value = 2;
            _suppressDataDirty = false;
            RebuildClassRows();
        }

        int classes = kind == "XOR" ? 2 : (_selectedAlgorithm == 0 ? 2 : (int)_classCount.Value);
        int perClass = (int)_samplesPerClass.Value;
        double noise = (double)_noise.Value;
        int seed = (int)_seed.Value;
        List<SamplePoint> generated = kind switch
        {
            "Linearly Separable" => DatasetGenerator.LinearlySeparable(classes, perClass, noise, seed),
            "XOR" => DatasetGenerator.Xor(perClass, noise, seed),
            "Spiral" => DatasetGenerator.Spiral(classes, perClass, noise, seed),
            _ => DatasetGenerator.Clusters(classes, perClass, noise, seed)
        };

        _suppressDataDirty = true;
        _plot.ClearAll();
        _plot.Samples.AddRange(generated);
        _suppressDataDirty = false;
        _datasetVersion++;
        _currentDataset = kind;
        _datasetName.Text = $"DATASET • {kind}";
        ResetActiveModel(clearLoss: true);
        _plot.Invalidate();
        UpdateSampleSummary();
        _splitSummary.Text = "Train: -   Test: -";
        Log($"Dataset oluşturuldu: {kind}, {generated.Count} örnek, {classes} sınıf, noise={noise:0.00}.");
        SetStatus($"{kind} veri kümesi hazır. Eğitimi başlatabilirsin.");
    }

    private void OnSamplesChanged()
    {
        if (_suppressDataDirty) return;
        MarkDatasetChanged("Veri noktaları değiştirildi. Mevcut model artık güncel değil.");
    }

    private void MarkDatasetChanged(string status)
    {
        _datasetVersion++;
        _plot.ModelStale = _plot.Classifier is not null;
        _plot.ClearEvaluation();
        _confusion.Matrix = null;
        ResetMetricCards();
        _splitSummary.Text = "Train: -   Test: -";
        foreach (var p in _plot.Samples) p.IsTest = false;
        UpdateSampleSummary();
        SetStatus(status);
    }

    private void ClearData()
    {
        _suppressDataDirty = true;
        _plot.ClearAll();
        _suppressDataDirty = false;
        _datasetVersion++;
        _runs.Clear();
        RefreshComparisonGrid();
        ResetActiveModel(clearLoss: true);
        _currentDataset = "Manual";
        _datasetName.Text = "DATASET • Manual";
        _splitSummary.Text = "Train: -   Test: -";
        UpdateSampleSummary();
        Log("Dataset ve karşılaştırma geçmişi temizlendi.");
        SetStatus("Grafiğe sol tıklayarak yeni örnek ekleyebilirsin.");
    }

    private async Task TrainAsync()
    {
        if (_plot.Samples.Count < 4)
        {
            MessageBox.Show("Önce en az 4 örnek ekle.", "Yetersiz veri", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        int classCount = _selectedAlgorithm == 0 ? 2 : (int)_classCount.Value;
        var present = _plot.Samples.Select(s => s.ClassIndex).Distinct().OrderBy(x => x).ToArray();
        if (!Enumerable.Range(0, classCount).All(present.Contains))
        {
            MessageBox.Show($"0..{classCount - 1} arasındaki her sınıftan en az bir örnek olmalı.", "Eksik sınıf", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var (trainPoints, evalPoints) = ApplyStratifiedSplit((int)_testPercent.Value, (int)_seed.Value, classCount);
        var trainX = trainPoints.Select(p => new[] { p.X, p.Y }).ToArray();
        var trainY = trainPoints.Select(p => p.ClassIndex).ToArray();
        var evalX = evalPoints.Select(p => new[] { p.X, p.Y }).ToArray();
        var evalY = evalPoints.Select(p => p.ClassIndex).ToArray();

        IProgressiveClassifier classifier = CreateClassifier();
        var snapshots = new List<ClassifierEpochSnapshot>();
        int every = Math.Max(1, (int)_epochs.Value / 45);

        SetBusy(true, $"{CurrentAlgorithmName()} eğitiliyor...");
        Log($"TRAIN START | model={CurrentAlgorithmName()} | train={trainPoints.Count} | test={evalPoints.Count} | lr={_learningRate.Value} | epoch={_epochs.Value} | momentum={_momentum.Value}");
        try
        {
            TrainingResult result = await Task.Run(() => classifier.FitWithProgress(
                trainX, trainY,
                snap => { if (snapshots.Count < 80) snapshots.Add(snap); },
                every));

            _lossChart.Values = result.ErrorHistory;
            if (_animateBoundary.Checked && snapshots.Count > 1)
                await AnimateSnapshotsAsync(snapshots, result.ErrorHistory);

            _plot.Classifier = classifier;
            _plot.ModelStale = false;
            _lossChart.DisplayedCount = 0;

            var evaluation = ClassificationEvaluator.Evaluate(classifier, evalX, evalY, classCount);
            var wrong = evalPoints.Where(p => classifier.Predict([p.X, p.Y]) != p.ClassIndex).ToArray();
            _plot.SetMisclassified(wrong);
            _confusion.Matrix = evaluation.ConfusionMatrix;
            UpdateMetricCards(result, evaluation);

            var run = new ModelRun
            {
                Id = ++_runCounter,
                Algorithm = CurrentAlgorithmName(),
                Classifier = classifier,
                Training = result,
                Evaluation = evaluation,
                DatasetVersion = _datasetVersion,
                TestPercent = (int)_testPercent.Value,
                Misclassified = wrong
            };
            _runs.Add(run);
            RefreshComparisonGrid();

            Log($"TRAIN END   | epoch={result.EpochsRun} | loss={result.FinalLoss:0.######} | test-acc={evaluation.Accuracy:P2} | macro-f1={evaluation.MacroF1:0.000}");
            SetStatus($"Eğitim tamamlandı • Test accuracy {evaluation.Accuracy:P2} • Macro F1 {evaluation.MacroF1:0.000}");
            _plot.RefreshDecisionVisuals();
        }
        catch (Exception ex)
        {
            Log("ERROR | " + ex.Message);
            MessageBox.Show(ex.Message, "Eğitim hatası", MessageBoxButtons.OK, MessageBoxIcon.Error);
            SetStatus("Eğitim başarısız.");
        }
        finally
        {
            SetBusy(false, null);
        }
    }

    private IProgressiveClassifier CreateClassifier()
    {
        double lr = (double)_learningRate.Value;
        int epochs = (int)_epochs.Value;
        double momentum = (double)_momentum.Value;
        int seed = (int)_seed.Value;
        return _selectedAlgorithm switch
        {
            0 => new BinaryPerceptron(lr, epochs, momentum, seed),
            1 => new MulticlassPerceptron(lr, epochs, momentum, seed),
            _ => new MlpClassifier((int)_hidden.Value, lr, epochs, momentum, seed)
        };
    }

    private (List<SamplePoint> Train, List<SamplePoint> Eval) ApplyStratifiedSplit(int testPercent, int seed, int classCount)
    {
        foreach (var p in _plot.Samples) p.IsTest = false;
        var rnd = new Random(seed);
        for (int c = 0; c < classCount; c++)
        {
            var group = _plot.Samples.Where(p => p.ClassIndex == c).OrderBy(_ => rnd.Next()).ToList();
            int testCount = group.Count <= 1 ? 0 : Math.Clamp((int)Math.Round(group.Count * testPercent / 100.0), 1, group.Count - 1);
            for (int i = 0; i < testCount; i++) group[i].IsTest = true;
        }

        var train = _plot.Samples.Where(p => !p.IsTest).ToList();
        var test = _plot.Samples.Where(p => p.IsTest).ToList();
        var eval = test.Count > 0 ? test : train;
        _splitSummary.Text = $"Train: {train.Count}   Test: {test.Count}";
        _plot.Invalidate();
        return (train, eval);
    }

    private async Task AnimateSnapshotsAsync(IReadOnlyList<ClassifierEpochSnapshot> snapshots, IReadOnlyList<double> history)
    {
        Log($"VIS | {snapshots.Count} gerçek epoch snapshot'ı ile karar sınırı animasyonu başlatıldı.");
        int maxFrames = 56;
        IEnumerable<ClassifierEpochSnapshot> frames = snapshots.Count <= maxFrames
            ? snapshots
            : snapshots.Where((_, i) => i % (int)Math.Ceiling(snapshots.Count / (double)maxFrames) == 0).Append(snapshots[^1]);

        foreach (var snap in frames)
        {
            _plot.Classifier = snap.Model;
            _plot.ModelStale = false;
            _epochCard.Value = snap.Epoch.ToString();
            _lossCard.Value = FormatLoss(snap.Loss);
            _lossChart.DisplayedCount = Math.Min(snap.Epoch, history.Count);
            SetStatus($"Epoch animasyonu • {snap.Epoch} • loss/error {FormatLoss(snap.Loss)}");
            _plot.Refresh();
            _lossChart.Refresh();
            await Task.Delay(34);
        }
    }

    private void ActivateSelectedRun()
    {
        if (_comparisonGrid.SelectedRows.Count == 0 || _comparisonGrid.SelectedRows[0].Tag is not ModelRun run)
        {
            MessageBox.Show("Önce karşılaştırma tablosundan bir model seç.", "Model seçimi", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }
        if (run.DatasetVersion != _datasetVersion)
        {
            MessageBox.Show("Bu model farklı bir dataset sürümünde eğitildi. Karar sınırı güvenle karşılaştırılamaz.", "Dataset değişti", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        _plot.Classifier = run.Classifier;
        _plot.ModelStale = false;
        _plot.SetMisclassified(run.Misclassified);
        _lossChart.Values = run.Training.ErrorHistory;
        _confusion.Matrix = run.Evaluation.ConfusionMatrix;
        UpdateMetricCards(run.Training, run.Evaluation);
        SelectAlgorithmByName(run.Algorithm);
        SetStatus($"Run #{run.Id} aktif • {run.Algorithm} • Accuracy {run.Evaluation.Accuracy:P2}");
    }

    private void SelectAlgorithmByName(string name)
    {
        int index = name.StartsWith("Binary", StringComparison.OrdinalIgnoreCase) ? 0 :
                    name.StartsWith("Multiclass", StringComparison.OrdinalIgnoreCase) ? 1 : 2;
        SelectAlgorithm(index, resetModel: false);
    }

    private void ResetActiveModel(bool clearLoss)
    {
        _plot.Classifier = null;
        _plot.ModelStale = false;
        _plot.ClearEvaluation();
        _confusion.Matrix = null;
        ResetMetricCards();
        if (clearLoss) _lossChart.Values = Array.Empty<double>();
        SetStatus("Aktif model sıfırlandı. Dataset korunuyor.");
    }

    private void UpdateMetricCards(TrainingResult training, ClassificationEvaluation evaluation)
    {
        _accuracyCard.Value = evaluation.Accuracy.ToString("P1");
        _f1Card.Value = evaluation.MacroF1.ToString("0.000");
        _precisionCard.Value = evaluation.MacroPrecision.ToString("0.000");
        _recallCard.Value = evaluation.MacroRecall.ToString("0.000");
        _epochCard.Value = training.EpochsRun.ToString();
        _lossCard.Value = FormatLoss(training.FinalLoss);
    }

    private void ResetMetricCards()
    {
        foreach (var c in new[] { _accuracyCard, _f1Card, _precisionCard, _recallCard, _epochCard, _lossCard }) c.Value = "-";
    }

    private void UpdateSampleSummary()
    {
        int count = _plot.Samples.Count;
        string byClass = string.Join("  ", Enumerable.Range(0, Math.Min((int)_classCount.Value, 5)).Select(c => $"C{c}:{_plot.Samples.Count(p => p.ClassIndex == c)}"));
        _sampleSummary.Text = $"{count} örnek  •  {byClass}";
    }

    private void ConfigureComparisonGrid()
    {
        _comparisonGrid.AllowUserToAddRows = false;
        _comparisonGrid.AllowUserToDeleteRows = false;
        _comparisonGrid.AllowUserToResizeRows = false;
        _comparisonGrid.ReadOnly = true;
        _comparisonGrid.RowHeadersVisible = false;
        _comparisonGrid.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
        _comparisonGrid.MultiSelect = false;
        _comparisonGrid.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
        _comparisonGrid.BackgroundColor = UiTheme.Plot;
        _comparisonGrid.BorderStyle = BorderStyle.None;
        _comparisonGrid.GridColor = UiTheme.Border;
        _comparisonGrid.DefaultCellStyle.BackColor = UiTheme.Plot;
        _comparisonGrid.DefaultCellStyle.ForeColor = UiTheme.Text;
        _comparisonGrid.DefaultCellStyle.SelectionBackColor = UiTheme.AccentSoft;
        _comparisonGrid.DefaultCellStyle.SelectionForeColor = UiTheme.Text;
        _comparisonGrid.ColumnHeadersDefaultCellStyle.BackColor = UiTheme.Surface2;
        _comparisonGrid.ColumnHeadersDefaultCellStyle.ForeColor = UiTheme.Muted;
        _comparisonGrid.ColumnHeadersDefaultCellStyle.SelectionBackColor = UiTheme.Surface2;
        _comparisonGrid.EnableHeadersVisualStyles = false;
        _comparisonGrid.Font = new Font("Segoe UI", 7.8f);
        _comparisonGrid.ColumnHeadersHeight = 30;
        _comparisonGrid.RowTemplate.Height = 27;
        _comparisonGrid.Columns.Add("Id", "#");
        _comparisonGrid.Columns.Add("Algorithm", "Model");
        _comparisonGrid.Columns.Add("Acc", "Acc");
        _comparisonGrid.Columns.Add("F1", "F1");
        _comparisonGrid.Columns.Add("Epoch", "Epoch");
        _comparisonGrid.Columns.Add("Loss", "Loss");
        _comparisonGrid.Columns[0].FillWeight = 22;
        _comparisonGrid.Columns[1].FillWeight = 90;
        _comparisonGrid.Columns[2].FillWeight = 55;
        _comparisonGrid.Columns[3].FillWeight = 50;
        _comparisonGrid.Columns[4].FillWeight = 48;
        _comparisonGrid.Columns[5].FillWeight = 60;
    }

    private void RefreshComparisonGrid()
    {
        _comparisonGrid.Rows.Clear();
        foreach (var run in _runs.OrderByDescending(r => r.Id))
        {
            int row = _comparisonGrid.Rows.Add(run.Id, ShortAlgorithm(run.Algorithm), run.Evaluation.Accuracy.ToString("P1"),
                run.Evaluation.MacroF1.ToString("0.000"), run.Training.EpochsRun, FormatLoss(run.Training.FinalLoss));
            _comparisonGrid.Rows[row].Tag = run;
            if (run.DatasetVersion != _datasetVersion)
                _comparisonGrid.Rows[row].DefaultCellStyle.ForeColor = Color.FromArgb(105, 125, 153);
        }
    }

    private void ConfigureLog()
    {
        _log.Text = "Decision Boundary Explorer\n" +
                    "Sol tık: yeni örnek / nokta sürükleme | Sağ tık: silme\n" +
                    "Train/test split ve tüm metrikler test seti üzerinden hesaplanır.\n\n";
    }

    private void Log(string text)
    {
        string line = $"[{DateTime.Now:HH:mm:ss}] {text}{Environment.NewLine}";
        _log.AppendText(line);
        _log.SelectionStart = _log.TextLength;
        _log.ScrollToCaret();
    }

    private void SetBusy(bool busy, string? status)
    {
        foreach (var c in _busySensitiveControls.Distinct())
            c.Enabled = !busy;
        _plot.Enabled = !busy;
        _statusProgress.Visible = busy;
        UseWaitCursor = busy;
        if (status is not null) SetStatus(status);
        if (!busy) UpdateAlgorithmUi();
    }

    private void SetStatus(string text) => _statusText.Text = text;

    private string CurrentAlgorithmName() => _selectedAlgorithm switch
    {
        0 => "Binary Perceptron",
        1 => "Multiclass Perceptron",
        _ => $"MLP ({(int)_hidden.Value} hidden)"
    };

    private static string ShortAlgorithm(string s) => s.StartsWith("Multiclass") ? "Multi Perc." : s.StartsWith("Binary") ? "Binary Perc." : "MLP";
    private static string FormatLoss(double value) => value >= 100 ? value.ToString("0") : value >= 1 ? value.ToString("0.###") : value.ToString("0.0000");
}
