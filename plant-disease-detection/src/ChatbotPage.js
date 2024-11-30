import React, { useState } from 'react';
import axios from 'axios';

const ChatbotPage = () => {
    const [query, setQuery] = useState('');
    const [chatResponse, setChatResponse] = useState('');
    const [followUpMessages, setFollowUpMessages] = useState([]); // Store follow-up messages

    const handleChatSubmit = async () => {
        if (!query) {
            setChatResponse('Please enter a query.');
            return;
        }

        try {
            const response = await axios.post('http://localhost:5000/chatbot', {
                message: query
            });

            setChatResponse(response.data.response); // Set the first part of the response
            setFollowUpMessages(response.data.followup); // Store follow-up messages

        } catch (error) {
            console.error('Error during chatbot communication:', error);
            setChatResponse('Error during communication with the chatbot.');
        }
    };

    const showFollowUpMessages = () => {
        // Append follow-up messages to the response display
        if (followUpMessages.length > 0) {
            setChatResponse(prevResponse => prevResponse + ' ' + followUpMessages.join(' '));
            setFollowUpMessages([]); // Clear follow-ups after displaying them
        }
    };

    return (
        <div style={{ textAlign: 'center', marginTop: '50px' }}>
            <h2>Chatbot Page</h2>
            <div style={{ width: '90%', maxWidth: '600px', margin: '0 auto' }}>
                <h3 style={{ fontSize: '30px', color: 'white'}}>Ask your queries</h3>
                <input
                    type="text"
                    placeholder="Ask your queries"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    style={{ width: '100%', padding: '10px', margin: '10px 0', borderRadius: '5px', border: '1px solid #ddd' }}
                />
                <button
                    onClick={handleChatSubmit}
                    style={{ padding: '10px 20px', borderRadius: '5px', backgroundColor: '#4CAF50', color: 'white', border: 'none', cursor: 'pointer' }}
                >
                    Ask
                </button>

                {chatResponse && (
                    <div style={{ marginTop: '20px', textAlign: 'left', fontSize: '18px', overflowY: 'auto', maxHeight: '400px', border: '1px solid #ddd', borderRadius: '5px', padding: '10px' }}>
                        {/* <h4 style={{ fontSize: '10px', color: 'white'}}>Bot Response:</h4> */}
                        <pre style={{ margin: '0', fontSize: 'inherit', whiteSpace: 'pre-wrap', wordWrap: 'break-word', color:'white'}}>{chatResponse}</pre>

                        {/* Display follow-up messages button if there are more parts */}
                        {followUpMessages.length > 0 && (
                            <button
                                onClick={showFollowUpMessages}
                                style={{ padding: '10px', marginTop: '10px', borderRadius: '5px', backgroundColor: '#007BFF', color: 'white', border: 'none', cursor: 'pointer' }}
                            >
                                Show More
                            </button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ChatbotPage;
