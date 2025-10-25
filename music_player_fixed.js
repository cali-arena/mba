import React, { useState, useContext, createContext } from 'react';
import {
  BarSongTitle,
  BottomBar,
  Button,
  PlayList,
  Song,
  SongTitle,
} from './styles.js';
import { songList } from './constants.js';

const buttonLabels = ['Not replaying', 'Replaying all', 'Replaying one'];

const PlayerContext = createContext();

const PlayerProvider = ({ children }) => {
  const [currentSongIndex, setCurrentSongIndex] = useState(null);
  const [replayMode, setReplayMode] = useState(buttonLabels[0]);

  const value = {
    currentSongIndex,
    setCurrentSongIndex,
    replayMode,
    setReplayMode,
  };

  return <PlayerContext.Provider value={value}>{children}</PlayerContext.Provider>;
};

const usePlayerContext = () => {
  const context = useContext(PlayerContext);
  if (!context) {
    throw new Error('usePlayerContext must be used within a PlayerProvider');
  }
  return context;
};

const ControlBar = () => {
  const { currentSongIndex, setCurrentSongIndex, replayMode, setReplayMode } = usePlayerContext();
  const currentSong = currentSongIndex !== null ? songList[currentSongIndex] : null;

  const handlePrevious = () => {
    if (currentSongIndex === null) return;
    
    if (replayMode === 'Not replaying') {
      if (currentSongIndex > 0) {
        setCurrentSongIndex(currentSongIndex - 1);
      }
    } else if (replayMode === 'Replaying all') {
      const newIndex = currentSongIndex === 0 ? songList.length - 1 : currentSongIndex - 1;
      setCurrentSongIndex(newIndex);
    } else if (replayMode === 'Replaying one') {
      // Stay on the same song in "Replaying one" mode
      setCurrentSongIndex(currentSongIndex);
    }
  };

  const handleNext = () => {
    if (currentSongIndex === null) return;
    
    if (replayMode === 'Not replaying') {
      if (currentSongIndex < songList.length - 1) {
        setCurrentSongIndex(currentSongIndex + 1);
      } else {
        setCurrentSongIndex(null);
      }
    } else if (replayMode === 'Replaying all') {
      const newIndex = currentSongIndex === songList.length - 1 ? 0 : currentSongIndex + 1;
      setCurrentSongIndex(newIndex);
    } else if (replayMode === 'Replaying one') {
      // Stay on the same song in "Replaying one" mode
      setCurrentSongIndex(currentSongIndex);
    }
  };

  const toggleReplayMode = () => {
    const currentIndex = buttonLabels.indexOf(replayMode);
    const nextIndex = (currentIndex + 1) % buttonLabels.length;
    setReplayMode(buttonLabels[nextIndex]);
  };

  return (
    <BottomBar>
      <BarSongTitle data-testid="barTitle">
        {currentSong ? `${currentSong.author} - ${currentSong.title}` : ''}
      </BarSongTitle>
      <div>
        <Button data-testid="previousButton" onClick={handlePrevious}>Previous</Button>
        <Button data-testid="nextButton" onClick={handleNext}>Next</Button>
        <Button data-testid="currentModeButton" onClick={toggleReplayMode}>{replayMode}</Button>
      </div>
    </BottomBar>
  );
};

const Songs = () => {
  const { currentSongIndex, setCurrentSongIndex } = usePlayerContext();

  return (
    <PlayList>
      {songList.map(({ title, author, id }, index) => (
        <Song 
          key={id}
          onClick={() => setCurrentSongIndex(index)}
          style={{ cursor: 'pointer' }}
        >
          <SongTitle 
            data-testid={id}
            active={currentSongIndex === index}
          >
            {title}
          </SongTitle>
          <p>{author}</p>
        </Song>
      ))}
    </PlayList>
  );
};

// Wrapper component to integrate Songs and ControlBar
const MusicPlayer = () => (
  <PlayerProvider>
    <Songs />
    <ControlBar />
  </PlayerProvider>
);

export { PlayerProvider, Songs, ControlBar, MusicPlayer };
