# MediaPipe ile Poz Tahmini
Mediapipe, Google tarafından oluşturulan makine öğrenimi çözümleri oluşturmak için kullandığımız açık kaynaklı bir frameworktür.
MediaPipe modüler yapısı sayesinde bize kullanımı kolay ve hızlı uygulanabilir bir yapı sunuyor. Bir çok platformda kullanılması da büyük bir avantaj sağlıyor. Biz bu çalışmamızda poz tahmini üzerinde duracağız. İsterseniz MediaPipe ile farklı uygulamalar da gerçekleştirebilirsiniz. Bunun için aşağıdaki linke bakabilirsiniz.

```bash
https://ai.google.dev/edge/mediapipe/solutions/guide?hl=tr
```

## Kurulum
**Repoyu klonlayın** <br>
```bash
git clone https://github.com/ensarakbas77/MediaPipe-PoseEstimation.git
cd MediaPipe-PoseEstimation
```
**Gerekli Kütüphaneler** <br>
```bash
pip install mediapipe
pip install opencv-python
```

**Modeli indirin ve çalışma klasörünüze taşıyın** <br> 
```bash 
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

### Örnek Görüntüler

*MediaPipe ile yapabileceğiniz çeşitli uygulama örnekleri:* <br> <br>
![General](https://github.com/user-attachments/assets/25552622-000e-4777-bb70-1ef1d5b7d07a) <br>

*MediaPipe ile görüntü ve video üzerinden poz tahmini örnekleri:* <br> <br>
<img width="300" height="300" alt="1" src="https://github.com/user-attachments/assets/7e9c943b-70cc-4fc0-8644-0856bf312a17" /> <br>
<img width="300" height="300" alt="2" src="https://github.com/user-attachments/assets/d860609f-445f-43d2-bf22-c8505c50c30a" />

