import React, { useEffect } from 'react';

const GoogleTranslate = () => {
    useEffect(() => {
        const addGoogleTranslateScript = () => {
            const script = document.createElement('script');
            script.type = 'text/javascript';
            script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
            script.onerror = () => console.error('Google Translate script failed to load.');
            document.body.appendChild(script);
        };

        window.googleTranslateElementInit = () => {
            try {
                if (window.google && window.google.translate) {
                    new window.google.translate.TranslateElement({
                        pageLanguage: 'en',
                        includedLanguages: 'ta,de,en,hi,bn,te,kn',
                        layout: window.google.translate.TranslateElement.InlineLayout.HORIZONTAL,
                        autoDisplay: false,
                        multilanguagePage: true,
                        gaTrack: true,
                        gaId: 'UA-XXXXX-Y'
                    }, 'google_translate_element');
                } else {
                    console.error('Google Translate API is not available.');
                }
            } catch (error) {
                console.error('An error occurred while initializing Google Translate:', error);
            }
        };

        addGoogleTranslateScript();
    }, []);

    // Inject CSS to hide unwanted Google Translate elements, including the top toolbar
    useEffect(() => {
        const style = document.createElement('style');
        style.innerHTML = `
            #table{
            display:none ;
            }
            .goog-logo-link {
                display: none !important;
            }
            .goog-te-gadget {
                font-size: 0 !important;
            }
            .goog-te-gadget .goog-te-combo {
                font-size: 14px !important;
                display: block !important;
                margin: 0 auto 20px auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            .goog-te-gadget .goog-te-combo:hover {
                border-color: #007BFF;
                box-shadow: 0 0 8px rgba(0,123,255,0.2);
            }
            .VIpgJd-ZVi9od-l4eHX-hSRGPd {
                display: none !important;
            }
            iframe.goog-te-banner-frame {
                display: none !important;
            }
            body {
                top: 0px !important;
            }
        `;
        document.head.appendChild(style);
        return () => {
            document.head.removeChild(style); // Cleanup on component unmount
        };
    }, []);

    return (
        <div className="google-translate-wrapper">
            <div className="child">
                <h2>Language:</h2>
                <div id="under"></div>
            </div>
            <div id="google_translate_element"></div>
        </div>
    );
};

export default GoogleTranslate;
