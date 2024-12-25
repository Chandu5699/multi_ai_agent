<template>
    <div>
      <h1>FastAPI Multimodal Inference</h1>
      <button @click="fetchInference">Run Inference</button>
      <div v-if="results">
        <h2>Results:</h2>
        <pre>{{ results }}</pre>
      </div>
      <div v-if="error">
        <h2>Error:</h2>
        <pre>{{ error }}</pre>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        results: null,
        error: null, // To handle errors
      };
    },
    methods: {
      async fetchInference() {
        try {
          // Post request to FastAPI inference endpoint
          const response = await this.$axios.post('/infer', ["sample.pdf", "audio.mp3"]);
          this.results = response.results; // Adjust based on FastAPI response structure
          this.error = null; // Reset error on success
        } catch (err) {
          this.error = err.response?.data || "An error occurred";
          console.error("Error fetching inference:", err);
        }
      },
    },
  };
  </script>
  
  <style scoped>
  h1 {
    color: #2c3e50;
  }
  button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    cursor: pointer;
  }
  button:hover {
    background-color: #2980b9;
  }
  </style>
  