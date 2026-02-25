document.getElementById("summarizeBtn").addEventListener("click", async () => {
    const text = document.getElementById("article").value.trim();
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = '';

    if (!text) {
        alert("Please paste a news article!");
        return;
    }

    resultDiv.innerHTML = '<div class="loading">Generating summaries...</div>';

    try {
        const response = await fetch("http://127.0.0.1:8000/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) throw new Error("Error: " + response.statusText);
        const data = await response.json();

        resultDiv.innerHTML = `
            <div class="summary-box">
                <h3>Extractive Summary</h3>
                <p>${data.extractive}</p>
            </div>
            <div class="summary-box">
                <h3>Abstractive Summary</h3>
                <p>${data.abstractive}</p>
            </div>
        `;

    } catch (err) {
        resultDiv.innerHTML = `<div style="color:red; text-align:center;">${err}</div>`;
    }
});