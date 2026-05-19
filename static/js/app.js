// =====================
// CropGuard AI — app.js
// =====================

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
let selectedFile = null;

// Drag and drop
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!allowed.includes(file.type)) {
    alert('Please upload a JPG, PNG, or WEBP image.');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    alert('File too large. Maximum size is 16MB.');
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('dropContent').style.display = 'none';
    document.getElementById('previewContent').style.display = 'block';
    dropZone.classList.add('has-image');
    detectBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function removeImage(e) {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = '';
  document.getElementById('dropContent').style.display = 'block';
  document.getElementById('previewContent').style.display = 'none';
  dropZone.classList.remove('has-image');
  detectBtn.disabled = true;
  document.getElementById('emptyState').style.display = 'flex';
  document.getElementById('resultsContent').style.display = 'none';
}

async function runDetection() {
  if (!selectedFile) return;

  const btn = document.getElementById('detectBtn');
  const btnText = btn.querySelector('.btn-text');
  const btnLoading = btn.querySelector('.btn-loading');

  btn.disabled = true;
  btnText.style.display = 'none';
  btnLoading.style.display = 'flex';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/detect', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    renderResults(data);

  } catch (err) {
    alert('Could not connect to server. Make sure Flask is running on port 5000.');
  } finally {
    btn.disabled = false;
    btnText.style.display = 'flex';
    btnLoading.style.display = 'none';
  }
}

function renderResults(data) {
  document.getElementById('emptyState').style.display = 'none';
  const content = document.getElementById('resultsContent');
  content.style.display = 'flex';

  // Disease info
  document.getElementById('diseaseName').textContent = data.disease;
  document.getElementById('diseaseCrop').textContent = 'Crop: ' + data.crop;
  document.getElementById('diseaseDesc').textContent = data.description;
  document.getElementById('confidenceVal').textContent = data.confidence + '%';

  // Severity badge
  const badge = document.getElementById('severityBadge');
  badge.textContent = data.severity === 'None' ? 'Healthy' : data.severity + ' Severity';
  badge.style.background = data.severity_color + '22';
  badge.style.color = data.severity_color;
  badge.style.border = '1px solid ' + data.severity_color + '44';

  // Treatment steps
  const steps = document.getElementById('treatmentSteps');
  steps.innerHTML = data.treatment.map((step, i) => `
    <div class="treatment-step">
      <div class="step-num">${i + 1}</div>
      <div class="step-text">${step}</div>
    </div>
  `).join('');

  // Prevention & organic
  document.getElementById('preventionText').textContent = data.prevention;
  document.getElementById('organicText').textContent = data.organic;

  // Top 3
  const top3 = document.getElementById('top3List');
  top3.innerHTML = data.top3.map((item, i) => `
    <div class="top3-item">
      <div class="top3-rank">${i + 1}</div>
      <div class="top3-name">${item.display}</div>
      <div class="top3-bar-wrap">
        <div class="top3-track">
          <div class="top3-fill" data-width="${item.confidence}" style="width:0%"></div>
        </div>
        <div class="top3-pct">${item.confidence}%</div>
      </div>
    </div>
  `).join('');

  // Animate top3 bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.top3-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 100);
  });

  if (window.innerWidth < 900) {
    content.scrollIntoView({ behavior: 'smooth' });
  }
}
