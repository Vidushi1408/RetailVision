import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post("http://127.0.0.1:8000/analyze_video/", formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });

    setResult(response.data.people_count);
  };

  return (
    <div>
      <h2>Retail AI System</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Analyze</button>
      {result !== null && <h3>People Detected: {result}</h3>}
    </div>
  );
}

export default App;
