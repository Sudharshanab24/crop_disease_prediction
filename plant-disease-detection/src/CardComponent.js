// CardComponent.js
import React from 'react';
import styled from 'styled-components';

const CardContainer = styled.div`
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 20px;
    margin: 20px;
`;

const Card = styled.div`
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    padding: 20px;
    width: 300px;
    text-align: center;
    transition: transform 0.3s ease;

    &:hover {
        transform: translateY(-10px);
    }
`;

const Image = styled.img`
    width: 100%;
    height: auto;
    border-radius: 8px;
    margin-bottom: 15px;
`;

const Title = styled.h3`
    font-size: 1.5em;
    margin-bottom: 10px;
`;

const Description = styled.p`
    font-size: 1em;
    color: #555;
`;

const CardComponent = () => {
    // Define your card data with images
    const cards = [
        {
            Plant_name:'APPLE',
            Disease_name: 'Apple_scab',
            District: 'Erode',
            image: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQhcwFBMaqjWO6gAW0Ss-dAcLDuvOgMw-rYpA&s'
        },
        {
            Plant_name: 'APPLE',
            Disease_name: 'Apple_blotch',
            District:'Erode',
            image: 'https://content.peat-cloud.com/w400/sooty-blotch-of-apple-apple-1574850811.jpg'
        },
        {
            Plant_name: 'APPLE',
            Disease_name: 'Apple_healthy',
            District:'Erode',
            image: 'https://tiimg.tistatic.com/fp/1/007/744/fresh-natural-no-added-preservatives-good-for-health-delicious-red-apple-944.jpg'
        },
        {
            Plant_name: 'APPLE',
            Disease_name: 'Apple_blotch',
            District:'Erode',
            image: 'https://content.peat-cloud.com/w400/sooty-blotch-of-apple-apple-1574850811.jpg'
        }
    ];

    return (
        <CardContainer>
            {cards.map((card, index) => (
                <Card key={index}>
                    <Image src={card.image} alt={card.title} />
                    <Title>{card.Plant_name}</Title>
                    <Description>{card.Disease_name}</Description>
                    <Description>{card.District}</Description>

                </Card>
            ))}
        </CardContainer>
    );
};

export default CardComponent;
