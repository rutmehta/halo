# file: websocket_client.py
"""
WebSocket client for testing real-time video face search.
This demonstrates how to continuously send video frames to the API
and receive similarity results in real-time.
"""

import asyncio
import websockets
import cv2
import json
import base64
import numpy as np
from datetime import datetime

async def video_search_client(uri: str, video_source=0):
    """
    Connect to the WebSocket endpoint and stream video frames.
    
    Args:
        uri: WebSocket URI (e.g., ws://localhost:8000/ws/video_search)
        video_source: Video source (0 for webcam, or path to video file)
    """
    # Open video capture
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Streaming video frames...")
        print("Press 'q' in the video window to quit.")
        
        frame_count = 0
        
        try:
            while True:
                # Read frame from video
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or error reading frame")
                    break
                
                # Display the frame locally
                cv2.imshow('Video Stream', frame)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Send frame to server
                await websocket.send(frame_bytes)
                
                # Receive and display results
                try:
                    # Set a timeout for receiving results
                    response = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=2.0
                    )
                    
                    result = json.loads(response)
                    
                    if result.get('query_face_found'):
                        print(f"\nFrame {frame_count}: Found similar faces!")
                        for i, match in enumerate(result.get('top_matches', [])):
                            print(f"  {i+1}. {match['image_url']} (score: {match['similarity_score']:.4f})")
                    else:
                        print(f"\nFrame {frame_count}: No face detected")
                        
                except asyncio.TimeoutError:
                    print(f"\nFrame {frame_count}: Timeout waiting for response")
                
                frame_count += 1
                
                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Add a small delay to control frame rate (30 FPS)
                await asyncio.sleep(0.033)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")

def test_with_static_images(uri: str, image_paths: list):
    """
    Test the WebSocket endpoint with static images instead of video.
    
    Args:
        uri: WebSocket URI
        image_paths: List of image file paths to test
    """
    async def send_images():
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            for i, image_path in enumerate(image_paths):
                print(f"\nTesting image {i+1}/{len(image_paths)}: {image_path}")
                
                # Read and encode image
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Send image
                await websocket.send(image_bytes)
                
                # Receive response
                response = await websocket.recv()
                result = json.loads(response)
                
                if result.get('query_face_found'):
                    print("Similar faces found:")
                    for j, match in enumerate(result.get('top_matches', [])):
                        print(f"  {j+1}. {match['image_url']} (score: {match['similarity_score']:.4f})")
                else:
                    print("No face detected in image")
                
                # Wait a bit between images
                await asyncio.sleep(1)
    
    asyncio.run(send_images())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WebSocket client for face search API')
    parser.add_argument(
        '--uri', 
        default='ws://localhost:8000/ws/video_search',
        help='WebSocket URI (default: ws://localhost:8000/ws/video_search)'
    )
    parser.add_argument(
        '--source',
        default=0,
        help='Video source: 0 for webcam, or path to video file (default: 0)'
    )
    parser.add_argument(
        '--images',
        nargs='+',
        help='Test with static images instead of video stream'
    )
    
    args = parser.parse_args()
    
    if args.images:
        # Test with static images
        test_with_static_images(args.uri, args.images)
    else:
        # Test with video stream
        # Convert source to int if it's a webcam index
        source = int(args.source) if args.source.isdigit() else args.source
        asyncio.run(video_search_client(args.uri, source)) 