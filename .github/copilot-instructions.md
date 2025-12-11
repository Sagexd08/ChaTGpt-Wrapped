# ChatGPT Wrapped - Development Instructions

## Project Overview
ChatGPT Wrapped is a full-stack analytics platform that processes ChatGPT conversation data and generates Spotify Wrapped-style recap cards with sentiment analysis, topic modeling, and shareable visuals.

## Tech Stack
- **Frontend:** Next.js 14, React, TailwindCSS, Framer Motion, Three.js
- **Backend:** Node.js with Express, Python analytics engine
- **Analytics:** Sentiment analysis, topic modeling, LLM integration
- **Image Generation:** AI-powered cover art generation
- **Export:** PNG export, MP4 (Phase 2)

## Setup Checklist
- [x] Create project structure
- [ ] Install dependencies
- [ ] Configure environment variables
- [ ] Set up database/storage
- [ ] Implement analytics engine
- [ ] Build frontend components
- [ ] Deploy and test

## Key Directories
- `/frontend` - Next.js application
- `/backend` - Node.js/Express services
- `/analytics` - Python data processing and analysis
- `/shared` - Shared utilities and types
- `/public` - Static assets

## Development Notes
- No user chat data is stored on servers (privacy first)
- Supports files up to 50MB with chunking
- Analytics processing target: <4 seconds
- Outputs are deterministic for regeneration
- Mobile-first design approach
