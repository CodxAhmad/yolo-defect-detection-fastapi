import React, { useState } from "react";
import { Routes, Route, Link } from "react-router-dom";
import { Container, Typography, Button, Card, CardContent, TextField } from "@mui/material";

function Home() {
  const [health, setHealth] = useState(null);

  const checkHealth = async () => {
    try {
      const res = await fetch("http://localhost:8000/health");
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      setHealth({ status: "error", detail: err.message });
    }
  };

  return (
    <Container sx={{ mt: 5 }}>
      <Typography variant="h3" gutterBottom>Defective Product Detection</Typography>
      <Typography variant="body1" gutterBottom>
        This app detects damaged parts of products from images using a YOLO model.
      </Typography>
      <Button variant="contained" color="primary" onClick={checkHealth} sx={{ mt: 2 }}>
        Check API Health
      </Button>
      {health && (
        <Card sx={{ mt: 2, p: 2, backgroundColor: "#f5f5f5" }}>
          <pre>{JSON.stringify(health, null, 2)}</pre>
        </Card>
      )}
      <Button variant="outlined" component={Link} to="/predict" sx={{ mt: 2 }}>
        Go to Predict
      </Button>
    </Container>
  );
}

function Predict() {
  const [file, setFile] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    setFile(e.target.files[0]);
    setImageURL(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Please select an image first!");
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const blob = await res.blob();
      setImageURL(URL.createObjectURL(blob));
    } catch (err) {
      alert("Prediction failed: " + err.message);
    }
    setLoading(false);
  };

  return (
    <Container sx={{ mt: 5 }}>
      <Typography variant="h4" gutterBottom>Upload Product Image</Typography>
      <Typography variant="body2" gutterBottom>
        Select an image of your product. Damaged parts will be highlighted after prediction.
      </Typography>
      <input type="file" accept="image/*" onChange={handleUpload} style={{ marginTop: "10px" }} />
      <Button variant="contained" onClick={handlePredict} sx={{ ml: 2 }} disabled={loading}>
        {loading ? "Processing..." : "Predict"}
      </Button>
      {imageURL && (
        <Card sx={{ mt: 3, p: 2 }}>
          <img src={imageURL} alt="Prediction" style={{ maxWidth: "100%" }} />
        </Card>
      )}
      <Button variant="outlined" component={Link} to="/" sx={{ mt: 2 }}>
        Back to Home
      </Button>
    </Container>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<Predict />} />
    </Routes>
  );
}