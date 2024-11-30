// UploadForm.js
import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import CardComponent from './CardComponent.js'; // Import the CardComponent

// Static Background Component
const StaticBackground = () => (
    <div
        style={{
            position: 'fixed', // Use fixed positioning to cover the entire viewport
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: -1, // Place behind other content
        }}
    />
);

// Styled components using framer-motion for animations
const Container = motion.div;
const Form = motion.form;
const Label = motion.label;
const Select = motion.select;
const Option = motion.option;
const Input = motion.input;
const Button = motion.button;

// Container for button to be at the top
const ButtonContainer = styled.div`
    position: fixed;
    top: 20px; // Adjust this value to your preference
    right: 20px; // Adjust this value to your preference
    z-index: 1;
`;

const UploadForm = () => {
    const [name, setName] = useState('');
    const [image, setImage] = useState(null);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    // Handle image file selection
    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const fileType = file.type;
            const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/svg+xml'];

            if (validImageTypes.includes(fileType)) {
                setImage(file);
                setError('');
            } else {
                setError('Unsupported file format. Please upload a JPEG, PNG, GIF, BMP, or SVG image.');
                setImage(null);
            }
        }
    };

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!name || !image) {
            setError('Name or image is missing');
            return;
        }

        const formData = new FormData();
        formData.append('name', name);
        formData.append('image', image);

        try {
            const response = await axios.post('http://127.0.0.1:5000/submit', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const { prediction, image_url } = response.data;
            navigate('/result', { state: { prediction, imageUrl: `http://127.0.0.1:5000${image_url}` } });
        } catch (error) {
            console.error('Error submitting form:', error.response ? error.response.data : error.message);
            setError('An error occurred while submitting the form.');
        }
    };

    return (
        <div style={{ position: 'relative', minHeight: '100vh' }}>
            <StaticBackground /> {/* Include the static background */}
            <ButtonContainer>
                <Button
                    onClick={() => navigate('/posts')} // Navigate to card component page
                    style={{ padding: '10px 20px', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}
                >
                    Posts
                </Button>
            </ButtonContainer>
            <Container
                initial={{ opacity: 0, y: -50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                style={{ position: 'relative', zIndex: 1, textAlign: 'center', marginTop: '100px', padding: '50px' }} // Adjust marginTop to avoid overlap with fixed button
            >
                <h2 style={{ fontSize: '50px', color: 'white'}}>GROOT</h2>
                <Form
                    onSubmit={handleSubmit}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    style={{
                        display: 'inline-block',
                        textAlign: 'left',
                        padding: '20px',
                        background: 'rgba(255, 255, 255, 0.8)', // Semi-transparent white background
                        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                        borderRadius: '10px',
                        border: 'none', // No border
                        position: 'relative',
                        zIndex: 2
                    }}
                >
                    <Label
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3 }}
                        htmlFor="name"
                        style={{ display: 'block', margin: '15px 0 5px', fontWeight: 'bold' }}
                    >
                        Name:
                    </Label>
                    <Select
                        id="name"
                        name="name"
                        required
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                        style={{ width: '90%', padding: '10px', margin: '5px 0 20px', border: '1px solid #ccc', borderRadius: '5px' }}
                    >
                        <Option value="" disabled>Select a crop</Option>
                        <Option value="Apple">Apple</Option>
                        <Option value="Potato">Potato</Option>
                        <Option value="Grapes">Grapes</Option>
                        <Option value="Corn">Corn</Option>
                        <Option value="Paddy">Paddy</Option>
                    </Select>
                    <Label
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3 }}
                        htmlFor="image"
                        style={{ display: 'block', margin: '15px 0 5px', fontWeight: 'bold' }}
                    >
                        Upload Image:
                    </Label>
                    <Input
                        type="file"
                        id="image"
                        name="image"
                        accept="image/*" // Accept all image types
                        required
                        onChange={handleImageChange}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                        style={{ width: '90%', padding: '10px', margin: '5px 0 20px', border: '1px solid #ccc', borderRadius: '5px' }}
                    />
                    {error && (
                        <p style={{ color: 'red', margin: '10px 0' }}>
                            {error}
                        </p>
                    )}
                    <Button
                        type="submit"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                        style={{ padding: '10px 20px', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}
                        whileHover={{ backgroundColor: '#0056b3', transform: 'translateY(-2px)' }}
                    >
                        Submit
                    </Button>
                </Form>
            </Container>
        </div>
    );
};

export default UploadForm;
