# Decision Boundary Explorer — Classification Laboratory

Bu ödev C# / .NET 8 Windows Forms ile hazırlanmıştır. Makine öğrenmesi algoritmaları hazır ML kütüphanesi kullanılmadan elle uygulanır.

## Algoritmalar
- Binary Perceptron
- Multiclass Perceptron
- Multi-Layer Perceptron (2 → hidden → class count)
- Z-Score normalization
- Momentum
- Sigmoid, Softmax, Cross-Entropy

## UI ve deney özellikleri
- Linearly Separable / XOR / Clusters / Spiral dataset üretimi
- Mouse ile örnek ekleme, sürükleme ve silme
- Sınıf rengi ve marker özelleştirme
- 2–5 sınıf
- Stratified train/test split
- Gerçek epoch snapshot'larıyla decision boundary animasyonu
- Decision region + sınır çizgisi
- Train/test örneklerini ayrı çizim stiliyle gösterme
- Yanlış sınıflandırmaları kırmızı halo ile gösterme
- Accuracy, Precision, Recall, Macro F1
- Confusion Matrix
- Loss/error history
- Aynı dataset üzerinde model karşılaştırma geçmişi
- Eski run modelini tekrar görselleştirme
- Model reset, aynı veriyle tekrar eğitim ve dataset reset
- Eğitim logu ve status/progress bar

## Çalıştırma
Visual Studio 2022'de **.NET Desktop Development** workload kurulu olmalıdır.

1. `ClassificationLaboratory.sln` dosyasını aç.
2. `ClassificationApp` projesini Startup Project yap.
3. F5 ile çalıştır.

## Demo önerisi
1. XOR dataset seç.
2. Önce Binary Perceptron eğit ve sonucu karşılaştırma listesinde bırak.
3. MLP sekmesine geç; 12 hidden neuron, 0.01 learning rate ve 0.90 momentum ile eğit.
4. Decision boundary ve Macro F1 farkını karşılaştır.
5. Spiral dataset üzerinde Multiclass Perceptron ve MLP'yi karşılaştır.

Bu akış lineer karar sınırı ile nonlinear MLP arasındaki farkı görsel olarak göstermeye uygundur.

## GitHub repo bilgileri

- **Depo adı:** `ClassificationLaboratory_ModernDashboard`
- **Platform:** Windows, .NET 8, Windows Forms
- **Solution:** `ClassificationLaboratory.sln`
- **Yapı:** Uygulama projesi + bağımsız `ML.Core` kaynak kodu

### Komut satırından derleme (Windows)

```powershell
dotnet restore ClassificationLaboratory.sln
dotnet build ClassificationLaboratory.sln --configuration Release
```

GitHub Actions, her push ve pull request için Windows runner üzerinde derleme kontrolü yapacak şekilde yapılandırılmıştır. **Bu depoya henüz otomatik test projesi eklenmemiştir.**

> Not: Lisans dosyası kaynak pakette bulunmadığı için otomatik olarak bir açık kaynak lisansı eklenmedi. Lisans koşulları depo sahibi tarafından belirlenmelidir.
