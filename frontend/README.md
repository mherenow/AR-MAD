# AI Image Classifier - Frontend

React-based web frontend for the AI Image Classifier system. This single-page application provides a user interface for uploading images and displaying classification results with Grad-CAM heatmap visualizations.

## Overview

The frontend is built with:
- **React 19** with TypeScript for type-safe component development
- **Vite** for fast development and optimized production builds
- **Vitest** and fast-check for unit and property-based testing

## Prerequisites

- Node.js 18+ and npm
- Backend API server running (see `../backend/README.md`)

## Setup

### 1. Install Dependencies

```bash
npm install
```

This installs all required dependencies including React, TypeScript, Vite, and testing libraries.

### 2. Environment Configuration

The frontend uses an environment variable to configure the API base URL:

- **`VITE_API_BASE_URL`**: Base URL for backend API requests (default: empty string for development proxy)

#### Development Mode (default)

During development, the frontend uses Vite's built-in proxy to forward API requests to the backend. No environment configuration is needed:

```bash
# No .env file needed - proxy handles routing
npm run dev
```

The proxy is configured in `vite.config.ts` to forward `/classify` and `/health` requests to `http://localhost:8000`.

#### Production Mode

For production builds where the frontend is served from a different origin than the backend, set the API base URL:

```bash
# .env.production
VITE_API_BASE_URL=https://api.example.com
```

Or set it at build time:

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

## Development

### Start Development Server

Start the Vite development server with hot module replacement:

```bash
npm run dev
```

**Expected output:**
```
  VITE v8.0.16  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

The application will be available at `http://localhost:5173`. Changes to source files will automatically reload in the browser.

**Note:** The backend API server must be running at `http://localhost:8000` for the application to function correctly. See `../backend/README.md` for backend setup instructions.

### Run Tests

Run the test suite once:

```bash
npm test
```

Run tests in watch mode (re-runs on file changes):

```bash
npm run test:watch
```

### Lint Code

Check code for style and potential issues:

```bash
npm run lint
```

## Production Build

### Create Production Build

Build the application for production:

```bash
npm run build
```

This command:
1. Runs TypeScript compiler (`tsc -b`) to check types
2. Builds optimized production bundle with Vite

**Output:** Static files are generated in the `dist/` directory.

### Preview Production Build

Test the production build locally:

```bash
npm run preview
```

This serves the `dist/` directory at `http://localhost:4173` (or another available port).

**Note:** The preview server does NOT include the development proxy. If your backend is not at the same origin, you must set `VITE_API_BASE_URL` before building:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── UploadForm.tsx   # Image upload and validation
│   │   └── ResultPanel.tsx  # Classification result display
│   ├── types/               # TypeScript type definitions
│   │   └── api.ts           # API response types
│   ├── utils/               # Utility functions
│   │   ├── api.ts           # API client (fetch wrapper)
│   │   └── validation.ts    # File validation logic
│   ├── test/                # Test setup and utilities
│   ├── App.tsx              # Root component
│   ├── main.tsx             # Application entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── dist/                    # Production build output (generated)
├── vite.config.ts           # Vite configuration
├── package.json             # Dependencies and scripts
└── README.md                # This file
```

## Usage

### Upload and Classify an Image

1. Click "Choose File" or drag-and-drop an image onto the upload area
2. Select an image file (JPEG, PNG, BMP, or WebP format, max 10MB)
3. Preview the selected image
4. Click "Classify Image" to submit
5. View the classification result:
   - **Label**: "FAKE" (red) or "REAL" (green)
   - **Confidence**: Percentage confidence in the classification
   - **Probabilities**: Separate probabilities for fake and real
   - **Heatmap**: Grad-CAM visualization showing important image regions

### Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- WebP (`.webp`)

### File Size Limits

- **Minimum**: 1 byte (empty files rejected)
- **Maximum**: 10 MB

### Error Handling

The frontend validates files before submission and displays error messages for:
- Unsupported file types
- Files too large (>10MB) or empty (0 bytes)
- Network errors
- Backend API errors (capacity limits, server errors)

## API Integration

The frontend communicates with the backend via two endpoints:

### `POST /classify`

Upload an image for classification.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Single file field named `image`

**Response (200 OK):**
```json
{
  "label": "FAKE",
  "confidence": 0.873,
  "prob_fake": 0.873,
  "prob_real": 0.127,
  "logit": 1.234,
  "cam_image_base64": "data:image/png;base64,iVBORw0KG..."
}
```

### `GET /health`

Check backend API health status.

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

See `src/utils/api.ts` for implementation details.

## Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `VITE_API_BASE_URL` | Base URL for API requests | `""` (empty, uses proxy) | `https://api.example.com` |

**Environment precedence:**
1. `.env.local` (not committed, highest priority)
2. `.env.production` (for production builds)
3. `.env.development` (for development)
4. `.env` (shared defaults)

## Troubleshooting

### "Network error" when submitting images

**Cause:** Backend API server is not running or not accessible.

**Solution:** Start the backend server at `http://localhost:8000`:
```bash
cd ../backend
python -m uvicorn main:app --reload
```

### "Service at capacity" error (503)

**Cause:** Backend is processing the maximum number of concurrent requests (4).

**Solution:** Wait a moment and try again. This is a temporary condition.

### Build fails with TypeScript errors

**Cause:** Type checking failed.

**Solution:** Fix TypeScript errors reported by the compiler:
```bash
npm run lint  # Check for issues
```

### Development server doesn't proxy requests

**Cause:** Backend URL is hardcoded or proxy configuration is incorrect.

**Solution:** Ensure `VITE_API_BASE_URL` is NOT set during development (unset or empty string). The proxy in `vite.config.ts` will handle routing.

## Development Notes

### Adding New Features

1. Create components in `src/components/`
2. Add types in `src/types/`
3. Update API client in `src/utils/api.ts` if adding new endpoints
4. Write tests alongside components (`.test.tsx` files)

### Testing Strategy

The project uses a dual testing approach:

- **Unit tests**: Specific scenarios and edge cases (Vitest + React Testing Library)
- **Property-based tests**: Universal properties across many random inputs (fast-check)

See `src/test/` for test utilities and `*.test.tsx` files for examples.

### Code Quality

The project uses:
- **ESLint**: Code linting with React-specific rules
- **TypeScript**: Static type checking
- **Strict mode**: React strict mode enabled for development

Run checks before committing:
```bash
npm run lint  # ESLint
npm test      # Test suite
npm run build # Type checking + build
```

## Production Deployment

### Static Hosting

The built frontend is a collection of static files that can be served from any web server or CDN:

```bash
# Build for production
npm run build

# Output directory: dist/
# Deploy dist/ to your hosting provider
```

**Hosting options:**
- Netlify, Vercel, Cloudflare Pages (automatic builds)
- AWS S3 + CloudFront
- Nginx, Apache (traditional web servers)

### Environment Configuration for Production

Set the backend API URL before building:

```bash
# Option 1: .env.production file
echo "VITE_API_BASE_URL=https://api.example.com" > .env.production
npm run build

# Option 2: Environment variable at build time
VITE_API_BASE_URL=https://api.example.com npm run build
```

### CORS Configuration

Ensure the backend's `ALLOWED_ORIGIN` environment variable includes your frontend's production URL:

```bash
# Backend .env
ALLOWED_ORIGIN=https://your-frontend-domain.com
```

See `../backend/README.md` for backend configuration details.

## Contributing

### Code Style

- Use TypeScript for type safety
- Follow React hooks best practices
- Keep components focused and single-purpose
- Write tests for new features
- Format code consistently (ESLint auto-fix: `npm run lint -- --fix`)

### Component Guidelines

- **Props**: Always define TypeScript interfaces
- **State**: Use React hooks (`useState`, `useEffect`)
- **Validation**: Perform client-side validation before API calls
- **Error handling**: Display user-friendly error messages
- **Loading states**: Show spinners during async operations

## License

[Add license information]

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review backend documentation (`../backend/README.md`)
3. Check the browser console for client-side errors
4. Check backend logs for server-side errors

---

**Quick Start:**
```bash
# Install and run (development)
npm install
npm run dev

# Build for production
npm run build
npm run preview
```
