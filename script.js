let model;
// !!! Ensure this list matches the order of your model's output classes !!!
const CLASSES = ["Realism", "Impressionism", "Imperialism", "Cubism", "Pop-Art", "Minimalism"];

async function loadModel() {
    try {
        // Loads model.json from the 'model' subfolder
        model = await tf.loadLayersModel('model/model.json');
        console.log("AnalyArt Engine Ready.");
        
        // Hide loader
        const loader = document.getElementById('loader-overlay');
        loader.style.opacity = '0';
        setTimeout(() => loader.style.display = 'none', 500);
    } catch (err) {
        console.error(err);
        alert("System Error: Use 'Live Server' in VS Code to load the AI model.");
    }
}

async function runInference() {
    const img = document.getElementById('imgPreview');
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.textContent = "Analyzing...";

    const predictions = tf.tidy(() => {
        // Convert image to tensor
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224]) // MobileNetV2 standard
            .toFloat();

        // MobileNetV2 Normalization: Scale pixels to [-1, 1]
        const offset = tf.scalar(127.5);
        return model.predict(tensor.sub(offset).div(offset).expandDims(0));
    });

    const data = await predictions.data();
    predictions.dispose();
    
    showUIResults(data);
    btn.disabled = false;
    btn.textContent = "Identify Style";
}

function showUIResults(data) {
    const section = document.getElementById('resultsSection');
    const status = document.getElementById('statusMessage');
    const list = document.getElementById('rankList');
    
    section.classList.remove('hidden');
    list.innerHTML = "";

    // Map probabilities to style names
    let results = CLASSES.map((name, i) => ({
        name: name,
        prob: Math.round(data[i] * 100)
    })).sort((a, b) => b.prob - a.prob);

    // Logic for "Not Recognized"
    if (results[0].prob < 20) {
        status.textContent = "Style not recognized in the AnalyArt system.";
        status.className = "status-box fail";
        document.getElementById('rankingsContainer').classList.add('hidden');
    } else {
        status.textContent = `Identified Movement: ${results[0].name}`;
        status.className = "status-box success";
        document.getElementById('rankingsContainer').classList.remove('hidden');

        results.forEach(res => {
            const html = `
                <div class="ranking-row">
                    <span class="style-label">${res.name}</span>
                    <div class="bar-bg"><div class="bar-fill" style="width: ${res.prob}%"></div></div>
                    <span class="percent-label">${res.prob}%</span>
                </div>`;
            list.insertAdjacentHTML('beforeend', html);
        });
    }
    section.scrollIntoView({ behavior: 'smooth' });
}

// UI Interaction
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    if (e.target.files.length) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = document.getElementById('imgPreview');
            img.src = event.target.result;
            img.style.display = "block";
            document.querySelector('.drop-zone__prompt').style.display = "none";
            document.getElementById('analyzeBtn').disabled = false;
        };
        reader.readAsDataURL(e.target.files[0]);
    }
};

document.getElementById('analyzeBtn').onclick = runInference;
document.getElementById('resetBtn').onclick = () => location.reload();

loadModel();