async function predict() {
    const data = {
        age: document.getElementById('age').value,
        sex: document.getElementById('sex').value,
        "chest pain type": document.getElementById('chestPain').value,
        "resting bp s": document.getElementById('bp').value,
        cholesterol: document.getElementById('cholesterol').value,
        "fasting blood sugar": document.getElementById('fbs').value,
        "resting ecg": document.getElementById('ecg').value,
        "max heart rate": document.getElementById('maxHR').value,
        "exercise angina": document.getElementById('angina').value,
        oldpeak: document.getElementById('oldpeak').value,
        "ST slope": document.getElementById('slope').value
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        document.getElementById('result').innerText =
            `Prediction: ${result.prediction === 1 ? "Heart Disease" : "No Heart Disease"}
            Probability: ${result.probability[1].toFixed(2)}`;
    } catch (error) {
        console.error("Error:", error);
    }
}
