# AI Interview System 🎤

An intelligent interview system built with Streamlit that provides real-time AI-powered interview assistance, security monitoring, and response authenticity detection.

## 🌟 Features

### Core Functionality
- **AI-Powered Interviews**: Dynamic question generation based on job position
- **Real-time Security Monitoring**: Face detection and prohibited object detection
- **Response Authenticity Detection**: AI-generated content detection with multiple providers
- **Multi-modal Input**: Support for both voice and text responses
- **Comprehensive Reporting**: Detailed interview analysis and recommendations

### Advanced Features
- **Eden AI Integration**: Professional AI detection using multiple engines
- **Performance Optimization**: Caching, memory management, and efficient processing
- **Real-time Feedback**: Instant analysis of interview responses
- **Security Compliance**: Environment monitoring and violation tracking
- **Export Capabilities**: CSV and JSON report generation

## 🚀 Quick Start

### Option 1: Streamlit Cloud Deployment (Recommended)

1. **Fork this repository** to your GitHub account

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Deploy!

3. **Configure API Keys**:
   - In Streamlit Cloud, go to your app settings
   - Add your OpenAI API key in the secrets section:
     ```toml
     [openai]
     api_key = "your-openai-api-key-here"
     ```

### Option 2: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/interview_bot_streamlit.git
   cd interview_bot_streamlit
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration**:
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   # Edit secrets.toml with your API keys
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Required API Keys

1. **OpenAI API Key** (Required):
   - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Used for interview question generation and response analysis

2. **Eden AI API Key** (Optional but recommended):
   - Get from [Eden AI](https://www.edenai.co/)
   - Provides professional AI detection with multiple engines
   - Significantly improves authenticity detection accuracy

### Environment Variables

You can configure the app using environment variables or Streamlit secrets:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
EDEN_AI_API_KEY=your-eden-ai-api-key
DEBUG_MODE=false
LOG_LEVEL=INFO
```

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Webcam (for security monitoring)
- Microphone (for voice input)

### Recommended for Best Performance
- Python 3.9+
- 8GB RAM
- Good internet connection
- Modern web browser

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🎯 Usage Guide

### For Interviewees

1. **Setup**:
   - Enter the position you're interviewing for
   - Provide your OpenAI API key in the interface
   - Enable camera and microphone permissions

2. **During the Interview**:
   - Position yourself in the camera frame
   - Respond to questions via voice or text
   - Follow security guidelines (no prohibited objects)

3. **After the Interview**:
   - Review your performance report
   - Download detailed analysis
   - Use recommendations for improvement

### For Interviewers/HR

1. **Configuration**:
   - Set up Eden AI for enhanced detection
   - Configure security monitoring settings
   - Customize interview stages and questions

2. **Monitoring**:
   - Real-time authenticity detection
   - Security violation tracking
   - Performance metrics monitoring

3. **Analysis**:
   - Comprehensive interview reports
   - Authenticity analysis
   - Candidate recommendations

## 🔒 Security & Privacy

### Data Protection
- No personal data stored permanently
- Interview recordings not saved
- API keys encrypted in transit
- Local processing when possible

### Security Monitoring
- Real-time face detection
- Prohibited object detection
- Environment compliance checking
- Violation logging and reporting

### AI Detection
- Multiple detection engines via Eden AI
- Built-in pattern recognition
- Confidence scoring
- False positive minimization

## 🛠️ Development

### Project Structure
```
interview_bot_streamlit/
├── app.py                 # Main application
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
├── .streamlit/           # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml.template
├── interview_reports/    # Generated reports
└── README.md
```

### Key Components
- **AITextDetector**: Advanced AI content detection
- **SecurityMonitor**: Real-time security monitoring
- **AIInterviewer**: Interview management and analysis
- **Performance Optimization**: Caching and memory management

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📊 Performance Optimizations

### Implemented Optimizations
- **Caching**: Streamlit caching for expensive operations
- **Memory Management**: LRU cache for AI detection results
- **Compiled Regex**: Pre-compiled patterns for better performance
- **Efficient Video Processing**: Reduced FPS and resolution options
- **API Rate Limiting**: Smart request management

### Monitoring
- Real-time performance metrics
- Memory usage tracking
- API call optimization
- Error rate monitoring

## 🚨 Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check browser permissions
   - Ensure camera is not used by other apps
   - Try refreshing the page

2. **Microphone issues**:
   - Check browser permissions
   - Test microphone in other applications
   - Consider using text input as alternative

3. **API errors**:
   - Verify API keys are correct
   - Check API rate limits
   - Ensure internet connection is stable

4. **Performance issues**:
   - Close other browser tabs
   - Reduce video quality in settings
   - Check system resources

### Getting Help
- Check the [Issues](https://github.com/yourusername/interview_bot_streamlit/issues) page
- Create a new issue with detailed description
- Include error messages and system information

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT API
- Eden AI for AI detection services
- Streamlit for the amazing framework
- MediaPipe for face detection
- YOLO for object detection

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with details
4. Contact the maintainers

---

**Made with ❤️ for better interviews**
