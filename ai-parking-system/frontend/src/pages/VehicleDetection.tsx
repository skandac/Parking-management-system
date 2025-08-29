import React, { useState, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  CameraAlt,
  Upload,
  PlayArrow,
  Stop,
  Save,
  Delete,
  Visibility,
} from '@mui/icons-material';

interface DetectionResult {
  bbox: [number, number, number, number];
  confidence: number;
  class_name: string;
  center: [number, number];
}

interface DetectionResponse {
  success: boolean;
  vehicles: DetectionResult[];
  total_vehicles: number;
  processing_time: number;
}

const VehicleDetection: React.FC = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResults, setDetectionResults] = useState<DetectionResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setDetectionResults(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleFileUpload = () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);

    // Simulate API call for vehicle detection
    setTimeout(() => {
      const mockResults: DetectionResponse = {
        success: true,
        vehicles: [
          {
            bbox: [100, 100, 300, 200],
            confidence: 0.95,
            class_name: 'car',
            center: [200, 150]
          },
          {
            bbox: [400, 150, 550, 250],
            confidence: 0.87,
            class_name: 'truck',
            center: [475, 200]
          }
        ],
        total_vehicles: 2,
        processing_time: 0.15
      };

      setDetectionResults(mockResults);
      setLoading(false);
    }, 2000);
  };

  const startVideoDetection = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            setIsDetecting(true);
            // Start detection loop
            detectFromVideo();
          }
        })
        .catch((err) => {
          setError('Error accessing camera: ' + err.message);
        });
    }
  };

  const stopVideoDetection = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsDetecting(false);
    }
  };

  const detectFromVideo = () => {
    if (!isDetecting || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    if (ctx && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // Simulate real-time detection
      const mockResults: DetectionResponse = {
        success: true,
        vehicles: [
          {
            bbox: [Math.random() * 400, Math.random() * 300, Math.random() * 200 + 100, Math.random() * 150 + 100],
            confidence: Math.random() * 0.3 + 0.7,
            class_name: ['car', 'truck', 'motorcycle'][Math.floor(Math.random() * 3)],
            center: [0, 0]
          }
        ],
        total_vehicles: 1,
        processing_time: Math.random() * 0.1 + 0.05
      };

      setDetectionResults(mockResults);
    }

    // Continue detection loop
    if (isDetecting) {
      setTimeout(detectFromVideo, 1000);
    }
  };

  const clearResults = () => {
    setDetectionResults(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        AI Vehicle Detection
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* File Upload Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Image Upload
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <Button
                variant="outlined"
                startIcon={<Upload />}
                onClick={() => fileInputRef.current?.click()}
                sx={{ mr: 2 }}
              >
                Select Image
              </Button>
              {selectedFile && (
                <Button
                  variant="contained"
                  onClick={handleFileUpload}
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <Visibility />}
                >
                  {loading ? 'Processing...' : 'Detect Vehicles'}
                </Button>
              )}
            </Box>

            {previewUrl && (
              <Box sx={{ textAlign: 'center' }}>
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ maxWidth: '100%', maxHeight: '300px', objectFit: 'contain' }}
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  {selectedFile?.name}
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Live Video Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Live Detection
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              {!isDetecting ? (
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<PlayArrow />}
                  onClick={startVideoDetection}
                >
                  Start Camera
                </Button>
              ) : (
                <Button
                  variant="contained"
                  color="secondary"
                  startIcon={<Stop />}
                  onClick={stopVideoDetection}
                >
                  Stop Camera
                </Button>
              )}
            </Box>

            <Box sx={{ position: 'relative', textAlign: 'center' }}>
              <video
                ref={videoRef}
                autoPlay
                muted
                style={{ 
                  width: '100%', 
                  maxHeight: '300px',
                  display: isDetecting ? 'block' : 'none'
                }}
              />
              <canvas
                ref={canvasRef}
                style={{ 
                  display: 'none',
                  position: 'absolute',
                  top: 0,
                  left: 0
                }}
              />
              
              {!isDetecting && (
                <Box sx={{ 
                  height: '300px', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  border: '2px dashed #ccc',
                  borderRadius: 1
                }}>
                  <Typography color="textSecondary">
                    Click "Start Camera" to begin live detection
                  </Typography>
                </Box>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Detection Results */}
        {detectionResults && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Detection Results
                </Typography>
                <Box>
                  <Tooltip title="Save Results">
                    <IconButton color="primary" sx={{ mr: 1 }}>
                      <Save />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Clear Results">
                    <IconButton color="error" onClick={clearResults}>
                      <Delete />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Chip 
                  label={`Total Vehicles: ${detectionResults.total_vehicles}`} 
                  color="primary" 
                  sx={{ mr: 1 }}
                />
                <Chip 
                  label={`Processing Time: ${detectionResults.processing_time.toFixed(3)}s`} 
                  color="secondary"
                />
              </Box>

              <Grid container spacing={2}>
                {detectionResults.vehicles.map((vehicle, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          {vehicle.class_name.charAt(0).toUpperCase() + vehicle.class_name.slice(1)}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Confidence: {(vehicle.confidence * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          BBox: [{vehicle.bbox.join(', ')}]
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Center: [{vehicle.center.join(', ')}]
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default VehicleDetection;
