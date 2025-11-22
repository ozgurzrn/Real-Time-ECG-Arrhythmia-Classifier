# Creating a Demo for GitHub

## Quick Demo GIF/Video

### Recommended Tools:
- **Windows**: ScreenToGif (free)
- **Mac**: Kap (free)
- **Cross-platform**: OBS Studio

### Steps to Record:

1. **Prepare** (2 min)
   - Have `sample_input.csv` ready
   - Launch Streamlit: `.venv_gpu\Scripts\streamlit run src/app.py`
   - Open browser to http://localhost:8502

2. **Record** (~30 seconds)
   - Show empty dashboard
   - Upload `sample_input.csv`
   - Show detection summary (with colored metrics)
   - Scroll to ECG visualization
   - Click dropdown and select different beats
   - Show Grad-CAM heatmap changing
   - Click "Download PDF Report" button
   - Show success message

3. **Export**
   - GIF: 800x600px, 10 FPS, max 10MB
   - Video: MP4, 1280x720, max 30 seconds

4. **Add to README**
   - Upload to `demo/` folder or use GitHub releases
   - Add to README: `![Demo](demo/demo.gif)`

### Alternative: Screenshots

If you prefer static images, create a carousel in README:

1. Empty dashboard
2. File uploaded + summary
3. ECG visualization with colored regions
4. Grad-CAM explanation
5. PDF download

Save as: `demo/screenshot_1.png`, etc.

Add to README:
```markdown
## Demo

<p align="center">
  <img src="demo/screenshot_1.png" width="45%" />
  <img src="demo/screenshot_2.png" width="45%" />
</p>
```
