"use client";

export default function ModelStatusBar({ useCloud, onToggleModel, modelStatus }) {
    return (
        <div className="model-bar">
            <div className="model-indicator">
                <span className={`model-dot ${useCloud ? "ready" : (modelStatus === "loading" ? "loading" : modelStatus === "ready" ? "ready" : "idle")}`} />
                <span>
                    Système : {useCloud ? "Gemini Neural-Net" : "Edge-GPU Inférence"}
                </span>
                <span style={{ opacity: 0.3, margin: '0 8px' }}>|</span>
                <span style={{ color: modelStatus === 'ready' || useCloud ? 'var(--poneglyph-glow)' : 'inherit' }}>
                    Statut : {useCloud ? "OPTIMAL" : modelStatus.toUpperCase()}
                </span>
            </div>
        </div>
    );
}
