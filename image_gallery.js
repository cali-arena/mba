import React, { useState, useEffect, useRef } from "react";

function ImageGallery({ images }) {
    const [loadedImages, setLoadedImages] = useState(new Set());
    const imageRefs = useRef([]);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        const index = parseInt(entry.target.dataset.index);
                        setLoadedImages(prev => new Set([...prev, index]));
                        observer.unobserve(entry.target);
                    }
                });
            },
            {
                rootMargin: '180px 0px 180px 0px' // Load when 180px below viewport
            }
        );

        // Observe all image containers
        imageRefs.current.forEach((ref) => {
            if (ref) {
                observer.observe(ref);
            }
        });

        return () => {
            observer.disconnect();
        };
    }, [images]);

    const galleryStyle = {
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 280px)',
        gap: '0px',
        justifyContent: 'center',
        padding: '20px'
    };

    const imageContainerStyle = {
        width: '280px',
        height: '280px',
        border: '1px solid #ccc',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#f5f5f5'
    };

    const imageStyle = {
        width: '280px',
        height: '280px',
        objectFit: 'cover'
    };

    return (
        <div style={galleryStyle}>
            {images.map((imageUrl, index) => (
                <div
                    key={index}
                    ref={(el) => {
                        imageRefs.current[index] = el;
                    }}
                    data-index={index}
                    style={imageContainerStyle}
                >
                    <img
                        src={loadedImages.has(index) ? imageUrl : ''}
                        alt={`Gallery image ${index + 1}`}
                        style={imageStyle}
                        onLoad={() => console.log(`Image ${index + 1} loaded`)}
                        onError={() => console.log(`Error loading image ${index + 1}`)}
                    />
                </div>
            ))}
        </div>
    );
}

export default ImageGallery;
