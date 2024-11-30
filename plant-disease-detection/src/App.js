import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import UploadForm from './UploadForm';
import ResultPage from './ResultPage';
import GoogleTranslate from './GoogleTranslate';
import ChatbotPage from './ChatbotPage';
import AnimatedBackground from './AnimatedBackground';
import CardComponent from './CardComponent';


const App = () => {
    return (
        <Router>
            <div style={{ position: 'relative', zIndex: 0 }}>
                <AnimatedBackground />
                <Routes>
                    {/* Show GoogleTranslate only on the UploadForm page */}
                    <Route path="/" element={<UploadForm />} />
                    <Route path="/result" element={<ResultPage />} />
                    <Route path="/posts" element={<CardComponent />} />
                    <Route path="/chatbot" element={<ChatbotPage />} />
                </Routes>
            </div>
        </Router>
    );
};

export default App;
