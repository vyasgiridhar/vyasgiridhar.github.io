// Constellation — subtle network effect, bottom of page
// Canvas 2D. Simple. Clean.

const PARTICLE_COUNT = 80;
const LINE_DIST = 120;
const MOUSE_RADIUS = 150;

export function initParticles(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return false;

  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const accent = isDark ? [94, 234, 212] : [13, 148, 136]; // teal

  let W, H, dpr;
  let mouseX = -9999, mouseY = -9999;

  function resize() {
    dpr = Math.min(window.devicePixelRatio, 2);
    W = window.innerWidth;
    H = 250;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // Particles
  const particles = [];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.3,
      r: 1.5 + Math.random() * 1.5,
    });
  }

  // Mouse — track relative to canvas position
  document.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
  });

  canvas.addEventListener('mouseleave', () => {
    mouseX = -9999;
    mouseY = -9999;
  });

  function tick() {
    ctx.clearRect(0, 0, W, H);

    // Update positions
    for (const p of particles) {
      // Mouse repulsion
      const dx = p.x - mouseX;
      const dy = p.y - mouseY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < MOUSE_RADIUS && dist > 0) {
        const force = (1 - dist / MOUSE_RADIUS) * 2;
        p.vx += (dx / dist) * force;
        p.vy += (dy / dist) * force;
      }

      p.x += p.vx;
      p.y += p.vy;

      // Damping
      p.vx *= 0.98;
      p.vy *= 0.98;

      // Wrap
      if (p.x < 0) p.x += W;
      if (p.x > W) p.x -= W;
      if (p.y < 0) p.y += H;
      if (p.y > H) p.y -= H;
    }

    // Draw lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < LINE_DIST) {
          const alpha = (1 - dist / LINE_DIST) * 0.25;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(${accent[0]},${accent[1]},${accent[2]},${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    // Draw dots
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${accent[0]},${accent[1]},${accent[2]},0.6)`;
      ctx.fill();
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
  return true;
}
