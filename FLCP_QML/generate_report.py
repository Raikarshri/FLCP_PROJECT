import json

def generate_html_report():
    # Load results
    with open("model_results.json", "r") as f:
        results = json.load(f)

    classical_accuracy = results["classical_accuracy"]
    quantum_accuracy = results["quantum_accuracy"]
    num_features = results["num_features"]
    test_size = results["test_size"]
    total_samples = results["total_samples"]

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Classification Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}

        .container {{
            max-width: 800px;
            width: 100%;
        }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .results-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}

        .result-card {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .result-card:hover {{
            transform: translateY(-5px);
        }}

        .result-card h2 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .accuracy-value {{
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
            border-radius: 10px;
            padding: 20px;
        }}

        .classical {{
            color: #667eea;
            background-color: #f0f4ff;
        }}

        .quantum {{
            color: #764ba2;
            background-color: #f8f3ff;
        }}

        .comparison {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            text-align: center;
        }}

        .comparison h2 {{
            color: #333;
            margin-bottom: 20px;
        }}

        .comparison-bar {{
            margin: 30px 0;
        }}

        .bar-label {{
            text-align: left;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }}

        .bar {{
            height: 40px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
        }}

        .classical-bar {{
            background: linear-gradient(90deg, #667eea 0%, #667eea {classical_accuracy*100}%, #e0e0e0 {classical_accuracy*100}%, #e0e0e0 100%);
        }}

        .quantum-bar {{
            background: linear-gradient(90deg, #764ba2 0%, #764ba2 {quantum_accuracy*100}%, #e0e0e0 {quantum_accuracy*100}%, #e0e0e0 100%);
        }}

        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.8;
            font-size: 0.9em;
        }}

        @media (max-width: 600px) {{
            .results-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🫁 Lung Cancer Classification</h1>
            <p>Classical SVM vs Quantum Machine Learning Comparison</p>
        </div>

        <div class="results-grid">
            <div class="result-card">
                <h2>Classical Model</h2>
                <p style="color: #666; margin-bottom: 15px;">Support Vector Machine (SVM)</p>
                <div class="accuracy-value classical">{classical_accuracy:.2%}</div>
                <p style="color: #999; margin-top: 10px;">Accuracy Score</p>
            </div>

            <div class="result-card">
                <h2>Quantum Model</h2>
                <p style="color: #666; margin-bottom: 15px;">VQC (Variational Quantum Classifier)</p>
                <div class="accuracy-value quantum">{quantum_accuracy:.2%}</div>
                <p style="color: #999; margin-top: 10px;">Accuracy Score</p>
            </div>
        </div>

        <div class="comparison">
            <h2>📊 Accuracy Comparison</h2>
            <div class="comparison-bar">
                <div class="bar-label">Classical (SVM)</div>
                <div class="bar classical-bar">{classical_accuracy:.2%}</div>

                <div class="bar-label">Quantum (VQC)</div>
                <div class="bar quantum-bar">{quantum_accuracy:.2%}</div>
            </div>

            <p style="color: #666; margin-top: 30px; line-height: 1.6;">
                <strong>Dataset:</strong> Lung Cancer Dataset<br>
                <strong>Features:</strong> {num_features} (PCA reduced)<br>
                <strong>Test Set Size:</strong> {test_size} samples ({test_size/total_samples*100:.0f}%)<br>
                <strong>Total Samples:</strong> {total_samples}<br>
                <strong>Classical Model:</strong> SVM with RBF kernel<br>
                <strong>Quantum Model:</strong> VQC with ZZFeatureMap & RealAmplitudes
            </p>
        </div>

        <div class="footer">
            <p>Generated by QML Model | Powered by Qiskit & Scikit-Learn</p>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML report
    with open("results.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("✅ HTML report generated: results.html")

if __name__ == "__main__":
    generate_html_report()