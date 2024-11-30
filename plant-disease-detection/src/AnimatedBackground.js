import React from 'react';
import { motion } from 'framer-motion';

const AnimatedBackground = () => {
    return (
        <motion.div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: -1, // Ensure it stays behind other content
                background: 'url(https://i.redd.it/k1ac1xpb31wy.png) no-repeat center center fixed',
                backgroundSize: 'cover',
                backgroundAttachment: 'fixed',
               
            }}
           
        />
    );
};

export default AnimatedBackground;
