import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import axios from 'axios';
import { FaRobot } from 'react-icons/fa';  // Icon for chatbot

const Container = styled.div`
    text-align: center;
    margin-top: 50px;
`;

const Image = styled.img`
    width: 150px;  // Increase the width
    height: 150px;  // Increase the height
    border: 1px solid #ddd;
    border-radius: 5px;
`;

const Button = styled.button`
    padding: 10px 20px;
    border-radius: 5px;
    background-color: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
`;

const ChatbotIcon = styled.div`
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #4CAF50;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: white;
    font-size: 24px;
`;

const Heading = styled.h2`
    font-size: 36px;  // Increase font size
`;

const SubHeading = styled.h3`
    font-size: 28px;  // Increase font size
`;

const Paragraph = styled.p`
    font-size: 20px;  // Increase font size
`;

const ResultPage = () => {
    const location = useLocation();
    const { prediction, imageUrl } = location.state || {};
    const navigate = useNavigate();  // Initialize the navigation hook

    const [query, setQuery] = useState('');
    const [chatResponse, setChatResponse] = useState('');

    const handleChatSubmit = async () => {
        try {
            const response = await axios.post('http://localhost:5000/chatbot', {
                message: query
            });
            setChatResponse(response.data.response);
        } catch (error) {
            console.error('Error during chatbot communication:', error);
            setChatResponse('Error during communication with the chatbot.');
        }
    };

    // Navigate to Chatbot Page on Icon Click
    const goToChatbot = () => {
        navigate('/chatbot', { state: { prediction } });  // Pass the prediction if needed
    };

    return (
        <Container>
            <Heading style={{ fontSize: '30px', color: 'white'}}>Result Page</Heading>
            {imageUrl && (
                <div>
                    <SubHeading>Uploaded Image:</SubHeading>
                    <Image src={imageUrl} alt="Uploaded" />
                </div>
            )}
            {prediction && (
                <div>
                    <SubHeading style={{ fontSize: '30px', color: 'white'}}>Predicted Disease:</SubHeading>
                    <Paragraph style={{ fontSize: '30px', color: 'white'}}>{prediction}</Paragraph>
                </div>
            )}

            {/* Floating Chatbot Icon/Button */}
            <ChatbotIcon onClick={goToChatbot}>
                <FaRobot />  {/* Robot icon for chatbot */}
            </ChatbotIcon>
        </Container>
    );
};

export default ResultPage;
