import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setMessages([{ text: 'Hello! How can I assist you today?', sender: 'bot' }]);
  }, []);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/predict', { message: input });
      const botMessage = { text: response.data.response, sender: 'bot' };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMessage = { text: 'Error communicating with the chatbot.', sender: 'bot' };
      setMessages((prev) => [...prev, errorMessage]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="app">
      <div className="container chatbot-container">
        <header className="header">
          <h1>Chat with Our Bot</h1>
          <button onClick={() => navigate('/')} className="back-button">
            Back to Tumor Analysis
          </button>
        </header>
        <div className="chat-window">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <p>{msg.text}</p>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={loading}
          />
          <button onClick={handleSendMessage} disabled={loading} className="analyze-button">
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;