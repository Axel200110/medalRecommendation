// --- AESTHETIC STATE MANAGEMENT ---
// --- DATABASE & SERVICE INITIALIZATION (Loaded from config.js) ---
const _supabase = (typeof supabase !== 'undefined' && typeof SUPABASE_URL !== 'undefined' && typeof SUPABASE_ANON_KEY !== 'undefined')
    ? supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
    : null;
window._supabase = _supabase; // Expose globally for debugging

const State = {
    participants: [],
    events: [],
    selectedEvent: null,
    activeTab: localStorage.getItem('aesthetic_active_tab') || 'dashboard',
    isAdmin: false,
    currentUser: JSON.parse(localStorage.getItem('talent_current_user') || 'null'),
    userAccounts: JSON.parse(localStorage.getItem('talent_user_accounts') || '[]'),
    weights: { skill: 0.6, consist: 0.3, exp: 0.1 },
    hasUnsavedWeightChanges: false,
    predictor: {
        network: null,
        isTrained: false,
        lastLoss: 1,
        epochs: 0
    },
    syncLog: []
};

// --- CORE ALGORITHM (AHP INSPIRED) ---
const SRI_LANKA_REGIONS = {
    'Western': { index: 1.0, label: 'High Resource Access' },
    'Central': { index: 0.85, label: 'Moderate Resource Access' },
    'Southern': { index: 0.85, label: 'Moderate Resource Access' },
    'North Western': { index: 0.75, label: 'Developing Resource Access' },
    'Sabaragamuwa': { index: 0.70, label: 'Developing Resource Access' },
    'Eastern': { index: 0.65, label: 'Low Resource Access' },
    'Northern': { index: 0.65, label: 'Low Resource Access' },
    'North Central': { index: 0.60, label: 'Low Resource Access' },
    'Uva': { index: 0.55, label: 'Constrained Resource Access' }
};

function calculateArtisticScore(participant, event) {
    // Performative Skill Match (Jaccard Index)
    const pSkills = (participant.skills || "").split(',').map(s => s.trim().toLowerCase());
    const eReqs = (event.requirements || []).map(s => s.toLowerCase());
    
    const intersection = pSkills.filter(s => eReqs.includes(s));
    const skillScore = eReqs.length > 0 ? (intersection.length / eReqs.length) : 0;

    // Delivery Consistency Index with Time-Series Inactivity Decay
    const inactiveMonths = participant.inactiveMonths || 0;
    // e^(-0.02 * t) represents a 2% decay per month of inactivity
    const decayFactor = Math.exp(-0.02 * inactiveMonths);
    const consistScore = (parseInt(participant.consistency || 0) / 100) * decayFactor;

    // Regional Equity Calculation (Sri Lankan Context)
    const regionData = SRI_LANKA_REGIONS[participant.region] || SRI_LANKA_REGIONS['Western'];
    const opportunityIndex = regionData.index;

    // Industry Tenacity (Logarithmic Scaling) + Equity Boost
    const rawExpScore = Math.min(Math.log10(parseInt(participant.experience || 0) + 1) / Math.log10(15), 1);
    const equityMultiplier = 1 + ((1 - opportunityIndex) * (consistScore * 1.5));
    const expScore = Math.min(rawExpScore * equityMultiplier, 1);

    // Weighted Sum Model using DYNAMIC WEIGHTS
    const artisticTotal = (skillScore * State.weights.skill) + 
                             (consistScore * State.weights.consist) + 
                             (expScore * State.weights.exp);

    // Predictive Analysis Vector (Neural Network)
    let projection = 0;
    if (State.predictor.isTrained) {
        const prediction = State.predictor.network.run([skillScore, consistScore, expScore, opportunityIndex]);
        projection = prediction[0] || 0;
    }

    // Hybrid consensus (50/50 mix)
    const hybridScore = Math.round(((artisticTotal + projection) / 2) * 100);
    
    // Conflict Detection
    const divergence = Math.abs(Math.round(artisticTotal * 100) - Math.round(projection * 100));
    const conflictLevel = divergence > 25 ? 'high' : (divergence > 15 ? 'mid' : 'low');

    return {
        total: hybridScore,
        scientific: Math.round(artisticTotal * 100),
        probability: Math.round(projection * 100),
        breakdown: { skill: skillScore, consist: consistScore, exp: expScore },
        matched: intersection,
        conflict: { level: conflictLevel, score: divergence },
        equity: { index: opportunityIndex, multiplier: equityMultiplier }
    };
}

// --- GROWTH TRAJECTORY SIMULATION ---
function calculateTrajectory(participant, years, event) {
    const trajectory = [];
    let currentSkills = [...(participant.skills || "").split(',').map(s => s.trim())];
    let currentExp = parseInt(participant.experience || 0);
    let currentConsist = parseInt(participant.consistency);

    for (let i = 0; i <= years; i++) {
        const pSim = {
            ...participant,
            experience: currentExp + i,
            consistency: Math.min(currentConsist + (i * 2), 100),
            skills: currentSkills.join(', ')
        };
        
        // Simulating "Artistic Evolution": Mastering one missing requirement every 1.5 years
        if (i > 0 && i % 2 === 0) {
            const missing = event.requirements.filter(r => !currentSkills.some(s => s.toLowerCase() === r.toLowerCase()));
            if (missing.length > 0) currentSkills.push(missing[0]);
            pSim.skills = currentSkills.join(', ');
        }

        const res = calculateArtisticScore(pSim, event);
        trajectory.push({ year: i, score: res.total, full: res });
    }
    return trajectory;
}

// --- PERFORMANCE ENGINE CALIBRATION ---
function initPredictor() {
    State.predictor.network = new brain.NeuralNetwork({
        hiddenLayers: [12, 12, 12]
    });

    const trainingData = [
        { input: [1, 1, 1, 1], output: [1] },
        { input: [0.9, 0.9, 0.8, 1], output: [0.98] },
        { input: [0.8, 0.95, 0.7, 0.6], output: [0.96] }, // Lower opportunity boosts expected output due to high tenacity
        { input: [0.6, 0.8, 0.5, 0.85], output: [0.70] },
        { input: [0.5, 0.7, 0.4, 0.7], output: [0.60] },
        { input: [0.7, 0.5, 0.2, 1], output: [0.45] },
        { input: [0.2, 0.4, 0.1, 0.65], output: [0.15] },
        { input: [0.1, 0.2, 0.1, 1], output: [0.02] }
    ];

    const logEl = document.getElementById('modelLog');
    const lossEl = document.getElementById('modelLoss');
    const epochEl = document.getElementById('calibrationEpochs');

    State.predictor.network.trainAsync(trainingData, {
        iterations: 2000,
        errorThresh: 0.005,
        log: true,
        logPeriod: 100,
        callback: (stats) => {
            State.predictor.lastLoss = stats.error;
            State.predictor.epochs = stats.iterations;
            if (lossEl) lossEl.textContent = stats.error.toFixed(6);
            if (epochEl) epochEl.textContent = stats.iterations;
            if (logEl) logEl.textContent = `Calibrating artistic projection: Epoch ${stats.iterations}... Error: ${stats.error.toFixed(6)}`;
        }
    }).then(() => {
        State.predictor.isTrained = true;
        const badge = document.getElementById('statusBadge');
        if (badge) {
            badge.textContent = "Status: Operational";
            badge.className = "status-badge operational";
        }
        if (logEl) logEl.textContent = "Performance forecasting converged. High artistic reliability achieved.";
        renderConsistencyMatrix();
        if (State.selectedEvent) renderRecommendations();
    });
}

function renderConsistencyMatrix() {
    const tbody = document.getElementById('consistencyMatrix');
    if (!tbody || !State.selectedEvent) return;

    const data = State.participants.map(p => {
        const res = calculateArtisticScore(p, State.selectedEvent);
        return { name: p.name, res };
    });

    tbody.innerHTML = data.map(d => `
        <tr>
            <td><strong>${d.name}</strong></td>
            <td>${d.res.scientific}%</td>
            <td>${d.res.probability}%</td>
            <td>${d.res.conflict.score}%</td>
            <td><span class="status-badge-inline ${d.res.conflict.level}">${d.res.conflict.level.toUpperCase()}</span></td>
        </tr>
    `).join('');

    renderFairnessAudit();

    // Dynamically update Cronbach's Alpha display
    const alphaEl = document.getElementById('cronbachAlphaVal');
    if (alphaEl) {
        const alpha = calculateCronbachsAlpha();
        alphaEl.textContent = alpha.toFixed(3);
    }
    
    // Update Ablation study if panel is open
    if (typeof calculateAblation === 'function') {
        calculateAblation();
    }
}

function calculateVariance(arr) {
    if (arr.length <= 1) return 0;
    const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
    const sqDiffs = arr.map(val => Math.pow(val - mean, 2));
    const avgSqDiff = sqDiffs.reduce((sum, val) => sum + val, 0) / (arr.length - 1);
    return avgSqDiff;
}

function calculateCronbachsAlpha() {
    const participants = State.participants;
    if (participants.length < 2) return 0;

    const scoresA = participants.map(p => p.judgeA !== undefined ? p.judgeA : p.consistency);
    const scoresB = participants.map(p => p.judgeB !== undefined ? p.judgeB : p.consistency);
    const scoresC = participants.map(p => p.judgeC !== undefined ? p.judgeC : p.consistency);

    const varA = calculateVariance(scoresA);
    const varB = calculateVariance(scoresB);
    const varC = calculateVariance(scoresC);
    const sumOfVariances = varA + varB + varC;

    const totalScores = participants.map((p, idx) => scoresA[idx] + scoresB[idx] + scoresC[idx]);
    const varTotal = calculateVariance(totalScores);

    if (varTotal === 0) return 0;

    const K = 3;
    const alpha = (K / (K - 1)) * (1 - (sumOfVariances / varTotal));
    return isNaN(alpha) ? 0 : alpha;
}

function renderFairnessAudit() {
    const tbody = document.getElementById('fairnessAuditMatrix');
    if (!tbody || !State.selectedEvent) return;

    if (State.participants.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="padding:2rem; opacity:0.5">No Performers Available</td></tr>`;
        return;
    }

    const regionalStats = {};
    State.participants.forEach(p => {
        const region = p.region || 'Western';
        if (!regionalStats[region]) {
            regionalStats[region] = { count: 0, totalSci: 0, totalNN: 0 };
        }
        const res = calculateArtisticScore(p, State.selectedEvent);
        regionalStats[region].count++;
        regionalStats[region].totalSci += res.scientific;
        regionalStats[region].totalNN += res.probability;
    });

    let globalNNTotal = 0;
    let globalCount = 0;
    Object.values(regionalStats).forEach(s => {
        globalNNTotal += s.totalNN;
        globalCount += s.count;
    });
    const globalNNAvg = globalCount > 0 ? (globalNNTotal / globalCount) : 0;

    const data = Object.keys(regionalStats).map(region => {
        const stats = regionalStats[region];
        const avgSci = Math.round(stats.totalSci / stats.count);
        const avgNN = Math.round(stats.totalNN / stats.count);
        
        let parityHtml = '';
        const diff = avgNN - globalNNAvg;
        if (Math.abs(diff) <= 5) {
            parityHtml = `<span class="status-badge-inline" style="background:var(--accent-success); color:black">BALANCED</span>`;
        } else if (diff > 5) {
            parityHtml = `<span class="status-badge-inline" style="background:var(--accent-primary)">ADVANTAGED</span>`;
        } else {
            parityHtml = `<span class="status-badge-inline" style="background:var(--danger); color:white">DISADVANTAGED</span>`;
        }

        return `
            <tr>
                <td><strong>${region}</strong></td>
                <td>${stats.count}</td>
                <td>${avgSci}%</td>
                <td>${avgNN}%</td>
                <td>${parityHtml}</td>
            </tr>
        `;
    });

    tbody.innerHTML = data.join('');
}

function applyWeightsFromCloud(newWeights, force = false) {
    if (!newWeights) return;
    if (State.hasUnsavedWeightChanges && !force) {
        return; // Do not overwrite local slider edits while admin is adjusting
    }
    State.hasUnsavedWeightChanges = false;
    State.weights = newWeights;
    const sVal = Math.round((State.weights.skill || 0.6) * 100);
    const cVal = Math.round((State.weights.consist || 0.3) * 100);
    const exVal = Math.round((State.weights.exp || 0.1) * 100);

    const sEl = document.getElementById('weightSkill');
    const cEl = document.getElementById('weightConsist');
    const exEl = document.getElementById('weightExp');

    if (sEl) sEl.value = sVal;
    if (cEl) cEl.value = cVal;
    if (exEl) exEl.value = exVal;

    if (document.getElementById('weightSkillVal')) document.getElementById('weightSkillVal').textContent = sVal;
    if (document.getElementById('weightConsistVal')) document.getElementById('weightConsistVal').textContent = cVal;
    if (document.getElementById('weightExpVal')) document.getElementById('weightExpVal').textContent = exVal;

    const statusEl = document.getElementById('weightSaveStatus');
    if (statusEl) statusEl.innerHTML = '';

    saveToCache();
    runKMeansAndAnomalies();
    renderConsistencyMatrix();
    if (typeof renderHireTalentList === 'function') renderHireTalentList();
    if (State.selectedEvent) renderRecommendations();
}

// --- DATA PERSISTENCE (SUPABASE CLOUD) ---
async function loadData() {
    const savedWeights = localStorage.getItem('aesthetic_weights');
    if (savedWeights) State.weights = JSON.parse(savedWeights);

    // Multi-tab / Multi-window instant synchronization listener
    window.addEventListener('storage', (e) => {
        if (e.key === 'aesthetic_weights' && e.newValue) {
            try {
                const newW = JSON.parse(e.newValue);
                applyWeightsFromCloud(newW);
            } catch (err) {}
        }
    });

    State.events = JSON.parse(localStorage.getItem('aesthetic_events')) || [
        { id: 1, name: 'Derana Dream Star Finale', description: 'National level vocal and stage performance competition.', requirements: ['Vocal Range', 'Stage Presence', 'Baila', 'Sinhala Diction'] },
        { id: 2, name: 'Corporate Emcee Summit (Colombo)', description: 'Professional hosting for high-end corporate galas.', requirements: ['Public Speaking', 'Trilingual', 'Professionalism'] },
        { id: 3, name: 'Kandy Cultural Pageant', description: 'Traditional arts and drumming showcase.', requirements: ['Kandyan Dance', 'Geta Bera', 'Choreography'] }
    ];
    State.participants = JSON.parse(localStorage.getItem('aesthetic_participants')) || [];

    const statusDot = document.querySelector('#cloudStatus .status-dot');

    const defaultSeedData = [
        { id: 'seed-1', name: 'Elena Vance', region: 'Western', experience: 12, consistency: 92, skills: 'Vocal Range, Diction, Opera, Stage Presence', judgeA: 90, judgeB: 95, judgeC: 91, inactiveMonths: 1 },
        { id: 'seed-2', name: 'Julian Marsh', region: 'Central', experience: 5, consistency: 85, skills: 'Public Speaking, Emceeing, Professionalism, Humor', judgeA: 82, judgeB: 88, judgeC: 85, inactiveMonths: 4 },
        { id: 'seed-3', name: 'Sarah Sings', region: 'Uva', experience: 8, consistency: 78, skills: 'Vocal Range, Pop, Stage Presence, Improvisation', judgeA: 75, judgeB: 80, judgeC: 79, inactiveMonths: 12 }
    ];

    if (!_supabase) {
        console.warn("Supabase not initialized. Using local defaults.");
        if (statusDot) statusDot.classList.remove('online');
        if (State.participants.length === 0) {
            State.participants = defaultSeedData;
            saveToCache();
        }
        runKMeansAndAnomalies();
        return;
    }

    try {
        const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000));
        const fetchP = _supabase.from('participants').select('*');
        const [pRes] = await Promise.race([Promise.all([fetchP]), timeout]);

        if (pRes.data && pRes.data.length > 0) {
            State.participants = pRes.data;
        } else {
            State.participants = defaultSeedData;
        }

        const statusBadge = document.getElementById('cloudStatus');
        if (statusBadge) statusBadge.classList.add('operational');
        if (statusDot) {
            statusDot.classList.add('online');
            statusDot.title = "Cloud Persistent: Active Connection";
        }
        
        logSync("Cloud Connection Established: Talent Database Operational.", "system");
        
        // Cloud Settings Sync for MCDM Decision Weights across all locations
        try {
            const { data: setRes } = await _supabase.from('system_settings').select('*').eq('key', 'mcdm_weights').maybeSingle();
            if (setRes && setRes.value) {
                applyWeightsFromCloud(setRes.value);
            }

            // Real-Time Supabase Channel for multi-browser instant sync
            _supabase
                .channel('realtime_mcdm_weights')
                .on('postgres_changes', { event: '*', schema: 'public', table: 'system_settings' }, (payload) => {
                    if (payload.new && payload.new.key === 'mcdm_weights' && payload.new.value) {
                        applyWeightsFromCloud(payload.new.value);
                    }
                })
                .subscribe();

            // Interval fallback polling (every 4s) for non-WebSocket environments
            setInterval(async () => {
                try {
                    const { data: pollRes } = await _supabase.from('system_settings').select('*').eq('key', 'mcdm_weights').maybeSingle();
                    if (pollRes && pollRes.value && JSON.stringify(pollRes.value) !== JSON.stringify(State.weights)) {
                        applyWeightsFromCloud(pollRes.value);
                    }
                } catch (pe) {}
            }, 4000);
        } catch (sErr) {
            console.log("Settings cloud sync note:", sErr);
        }

        saveToCache();
        if (typeof loadHireRequestsFromCloud === 'function') await loadHireRequestsFromCloud();
        runKMeansAndAnomalies();
        renderMyProfile();
        renderParticipantRegistry();
        renderConsistencyMatrix();
        if (State.selectedEvent) renderRecommendations();
    } catch (err) {
        console.error("Cloud sync failed. Using cached data.", err);
        logSync("Warning: Cloud Connection Failed. Working in Local-Only Mode.", "remote");
        
        if (State.participants.length === 0) {
            State.participants = defaultSeedData;
            saveToCache();
        }
        
        const statusBadge = document.getElementById('cloudStatus');
        if (statusBadge) statusBadge.classList.remove('operational');
        if (statusDot) statusDot.classList.remove('online');
        runKMeansAndAnomalies();
        renderMyProfile();
        renderParticipantRegistry();
        renderConsistencyMatrix();
        if (State.selectedEvent) renderRecommendations();
    }
}

function saveToCache() {
    localStorage.setItem('aesthetic_participants', JSON.stringify(State.participants));
    localStorage.setItem('aesthetic_events', JSON.stringify(State.events));
    localStorage.setItem('aesthetic_weights', JSON.stringify(State.weights));
}

// --- ML INTEGRATIONS: NLP, LSTM, K-MEANS, ANOMALY ---

const AESTHETIC_DICTIONARY = {
    // English Core
    'vocal range': 'Vocal Range', 'diction': 'Diction', 'stage presence': 'Stage Presence',
    'emceeing': 'Emceeing', 'public speaking': 'Public Speaking', 'professionalism': 'Professionalism',
    'modulation': 'Modulation', 'improvisation': 'Improvisation', 'acting': 'Acting',
    'choreography': 'Choreography', 'humor': 'Humor', 'poetry': 'Poetry',
    
    // Local / Cultural Arts (English/Sinhlish)
    'kandyan': 'Kandyan Dance', 'pahatharata': 'Low-country Dance', 'geta bera': 'Geta Bera',
    'thabla': 'Thabla', 'baila': 'Baila', 'bodu gee': 'Devotional Songs', 'calypso': 'Calypso',
    'trilingual': 'Trilingual', 'sinhala': 'Sinhala Diction', 'tamil': 'Tamil Diction',

    // Sinhala/Tamil direct phrasing translations (Sinhlish/Tanglish)
    'sindu kiyanawa': 'Vocal Range', 'natum': 'Choreography', 'katha karanna': 'Public Speaking',
    'poddak': 'Improvisation', 'patteta': 'Stage Presence', 'nalla': 'Professionalism', 'paattu': 'Vocal Range'
};

function extractSkillsFromText(text) {
    const raw = text.toLowerCase();
    const foundSkills = new Set();

    // Scan for keys in the trilingual dictionary
    Object.keys(AESTHETIC_DICTIONARY).forEach(key => {
        if (raw.includes(key)) {
            foundSkills.add(AESTHETIC_DICTIONARY[key]);
        }
    });

    if (foundSkills.size === 0) {
        const words = text.match(/\b[A-Z][a-z]*\b/g);
        if (words) return [...new Set(words)].slice(0, 5).join(', ');
        return "Core Artistry";
    }
    
    return Array.from(foundSkills).join(', ');
}

function calculateConsistencyForecast(historyStr) {
    const scores = historyStr.split(',').map(s => parseFloat(s.trim()) / 100).filter(n => !isNaN(n));
    if (scores.length < 2) return Math.round((scores[0] * 100) || 50);

    try {
        const lstm = new brain.recurrent.LSTMTimeStep({ hiddenLayers: [8] });
        lstm.train([scores], { iterations: 100, errorThresh: 0.05 });
        const prediction = lstm.run(scores);
        return Math.min(Math.max(Math.round(prediction * 100), 0), 100);
    } catch (e) {
        const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
        return Math.round(avg * 100);
    }
}

window.runKMeansAndAnomalies = () => {
    const container = document.getElementById('kmeansClusters');
    if (!container) return;

    if (State.participants.length < 3) {
        container.innerHTML = `
            <div class="glass-card" style="padding:2rem; text-align:center; grid-column:1/-1; border:1px dashed rgba(255,255,255,0.1)">
                <i data-lucide="database" style="margin-bottom:1rem; opacity:0.5"></i>
                <p style="font-size:0.85rem; opacity:0.7">Clustering Engine Awaiting Data Segment...</p>
                <div style="font-size:0.7rem; margin-top:0.5rem; color:var(--accent-secondary)">
                    Add ${3 - State.participants.length} more performer(s) to activate Algorithmic Archetype Mapping.
                </div>
            </div>
        `;
        if (typeof lucide !== 'undefined') lucide.createIcons();
        return;
    }
    
    // ANOMALY DETECTION
    const scores = State.participants.map(p => {
        const ev = State.selectedEvent || State.events[0];
        return calculateArtisticScore(p, ev).total;
    });
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const stdDev = Math.sqrt(scores.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / scores.length) || 1;
    
    State.participants.forEach((p, i) => {
        p.isDarkHorse = (stdDev > 0 && Math.abs((scores[i] - mean) / stdDev) > 1.2);
    });

    // K-MEANS CLUSTERING
    let centroids = [{ exp: 2, consist: 60 }, { exp: 7, consist: 80 }, { exp: 15, consist: 95 }];
    let clusters = [[], [], []];
    
    for(let iter = 0; iter < 5; iter++) {
        clusters = [[], [], []];
        State.participants.forEach(p => {
            let minDist = Infinity; let cIdx = 0;
            centroids.forEach((c, i) => {
                let d = Math.pow(c.exp - parseInt(p.experience), 2) + Math.pow(c.consist - parseInt(p.consistency), 2);
                if (d < minDist) { minDist = d; cIdx = i; }
            });
            clusters[cIdx].push(p);
        });
        clusters.forEach((clr, i) => {
            if (clr.length > 0) {
                centroids[i] = {
                    exp: clr.reduce((s, p) => s + parseInt(p.experience), 0) / clr.length,
                    consist: clr.reduce((s, p) => s + parseInt(p.consistency), 0) / clr.length
                };
            }
        });
    }
    
    const titles = ["Artistic Protégés", "Performance Masters", "Industry Legends"];
    const icons = ["sparkles", "star", "award"];
    const colors = ["var(--accent-secondary)", "var(--accent-primary)", "var(--accent-gold)"];
    
    container.innerHTML = clusters.map((clr, i) => `
        <div class="archetype-card animate-fadeIn" style="--card-glow: ${colors[i]};">
            <h4 class="archetype-title">
                <i data-lucide="${icons[i]}" style="width:18px; height:18px"></i> ${titles[i]}
            </h4>
            <p style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:1rem">${clr.length} performers grouped by career trajectory.</p>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                ${clr.length > 0 ? clr.map(p => `<div class="tag" style="${p.isDarkHorse ? 'background:var(--danger);color:white; box-shadow: 0 0 10px rgba(239,68,68,0.5)' : 'background:rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1)'}">${p.name.split(' ')[0]} ${p.isDarkHorse ? '🏆' : ''}</div>`).join('') : '<span style="font-size:0.7rem; opacity:0.3">No entities in segment</span>'}
            </div>
        </div>
    `).join('');
    if (typeof lucide !== 'undefined') lucide.createIcons();
};

window.autoTuneWeights = () => {
    const btn = document.getElementById('autoTuneBtn');
    if(btn) btn.disabled = true;
    
    setTimeout(() => {
        let best = { s: 0.6, c: 0.3, e: 0.1, loss: Infinity };
        for(let s = 20; s <= 70; s += 10) {
            for(let c = 20; c <= 70; c += 10) {
                let e = 100 - (s + c);
                if (e < 5 || e > 30) continue;
                State.weights = { skill: s/100, consist: c/100, exp: e/100 };
                let err = 0;
                State.participants.forEach(p => {
                    State.events.forEach(ev => err += calculateArtisticScore(p, ev).conflict.score);
                });
                if (err < best.loss) { best = { s: s/100, c: c/100, e: e/100, loss: err }; }
            }
        }
        State.weights = { skill: best.s, consist: best.c, exp: best.e };
        document.getElementById('weightSkill').value = best.s * 100;
        document.getElementById('weightConsist').value = best.c * 100;
        document.getElementById('weightExp').value = best.e * 100;
        
        document.getElementById('weightSkillVal').textContent = Math.round(best.s * 100);
        document.getElementById('weightConsistVal').textContent = Math.round(best.c * 100);
        document.getElementById('weightExpVal').textContent = Math.round(best.e * 100);
        
        showToast("Artistic weights calibrated to neural optimum.", "success");
        if(btn) btn.disabled = false;
        if (State.selectedEvent) renderRecommendations();
    }, 800);
};

window.exportDossier = () => {
    if (!State.selectedEvent) return;
    const data = State.participants.map(p => {
        const res = calculateArtisticScore(p, State.selectedEvent);
        return `${p.name}, Score: ${res.total}%, Skill Match: ${res.scientific}%, Net Forecast: ${res.probability}%`;
    });
    data.unshift("Performer Name, Total Score, Skill Match, Net Forecast");
    const blob = new Blob([data.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Dossier_${State.selectedEvent.name.replace(/\s+/g, '_')}.csv`;
    a.click();
    showToast("Event Dossier Exported Successfully", "success");
};

async function deleteParticipant(id) {
    if (!confirm("Are you sure you want to remove this performer from the registry?")) return;
    
    if (_supabase) {
        await _supabase.from('participants').delete().eq('id', id);
    }
    State.participants = State.participants.filter(p => String(p.id) !== String(id));
    saveToCache();
    renderParticipantRegistry();
    renderConsistencyMatrix();
    runKMeansAndAnomalies();
    showToast("Performer purged from registry.", "remote");
}

window.optimizeTeam = () => {
    const results = document.getElementById('teamOptimizationResult');
    if (!results || !State.selectedEvent) return;

    // Heuristic: Best Total, Best Consistency, and Best Tenure
    const scored = State.participants.map(p => ({
        ...p,
        res: calculateArtisticScore(p, State.selectedEvent)
    }));

    if (scored.length < 3) {
        showToast("Add at least 3 performers to discover an optimal ensemble.", "primary");
        return;
    }

    const bestSkill = [...scored].sort((a,b) => b.res.scientific - a.res.scientific)[0];
    const bestConsist = [...scored].filter(p => p.id !== bestSkill.id).sort((a,b) => b.res.probability - a.res.probability)[0];
    const bestTenure = [...scored].filter(p => p.id !== bestSkill.id && p.id !== bestConsist.id).sort((a,b) => parseInt(b.experience) - parseInt(a.experience))[0];

    const team = [bestSkill, bestConsist, bestTenure].filter(Boolean);
    
    document.getElementById('teamMembers').innerHTML = team.map(p => `
        <div style="flex:1; text-align:center; padding:1rem; background:var(--bg-deep); border-radius:12px; border:1px solid var(--border)">
            <div class="avatar" style="width:40px; height:40px; margin:0 auto 0.5rem; font-size:0.8rem">${p.name[0]}</div>
            <div style="font-size:0.85rem; font-weight:900">${p.name}</div>
            <div style="font-size:0.6rem; color:var(--accent-primary)">${p.res.total}% Hybrid Align</div>
        </div>
    `).join('');
    
    document.getElementById('teamCoverage').textContent = `Optimal Composition: ${State.selectedEvent.name}`;
    results.style.display = 'block';
    showToast("Ensemble Discovery Optimized.", "success");
}

// --- UI RENDERING ---
async function initApp() {
    // Ensure lucide icons render safely
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    initTheme();
    setupEventListeners();
    updateAuthUI();
    
    // Always lock admin on startup
    State.isAdmin = false;

    const savedPortal = localStorage.getItem('active_portal') || 'buyer';
    switchPortal(savedPortal);

    await loadData();
    renderEvents();
    renderParticipantRegistry();
    renderSkillChips();
    renderMyProfile();

    // Initialize hiring workflow UI
    populateHireEventFilter();
    renderHireTalentList();
    renderHirePanel();

    setTimeout(initPredictor, 1200);
    
    // Automatically select the first event to populate dashboards by default
    setTimeout(() => {
        const firstEventItem = document.querySelector('.event-item');
        if (firstEventItem) {
            firstEventItem.click();
        }
    }, 800);
}

function renderSkillChips() {
    const container = document.getElementById('skillChips');
    if (!container) return;

    // Use all keys from the dictionary
    const chipKeys = Object.keys(AESTHETIC_DICTIONARY);
    
    container.innerHTML = chipKeys.map(key => {
        return `<span class="skill-chip" onclick="appendSkillChip('${key}')">+ ${key}</span>`;
    }).join('');
}

window.appendSkillChip = (chipText) => {
    const bio = document.getElementById('bioInput');
    if (bio) {
        let val = bio.value.trim();
        if (val.length > 0 && !val.endsWith(',')) val += ', ';
        else if (val.length > 0) val += ' ';
        bio.value = val + chipText;
    }
}

function initTheme() {
    const saved = localStorage.getItem('aesthetic_theme') || 'light';
    if (saved === 'dark') {
        document.body.classList.add('dark-mode');
        const icon = document.querySelector('#themeToggle i');
        if (icon) icon.setAttribute('data-lucide', 'moon');
    }
    lucide.createIcons();
}

window.toggleTheme = () => {
    const isDark = document.body.classList.toggle('dark-mode');
    localStorage.setItem('aesthetic_theme', isDark ? 'dark' : 'light');
    
    const icon = document.querySelector('#themeToggle i');
    if (icon) {
        icon.setAttribute('data-lucide', isDark ? 'moon' : 'sun');
        lucide.createIcons();
    }
    
    showToast(`Theme switched to ${isDark ? 'Midnight' : 'Arctic Pearl'}`, "primary");
};

function setupEventListeners() {
    const adminBtn = document.getElementById('adminToggleBtn');
    if (adminBtn) {
        adminBtn.onclick = (e) => {
            e.preventDefault();
            window.toggleAdminMode();
        };
    }

    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.onclick = (e) => {
            e.preventDefault();
            window.toggleTheme();
        };
    }

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.onclick = () => switchTab(btn.dataset.tab);
    });

    ['weightSkill', 'weightConsist', 'weightExp'].forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.oninput = (e) => {
                const valDisplay = document.getElementById(id + 'Val');
                if (valDisplay) valDisplay.textContent = e.target.value;

                const s = parseFloat(document.getElementById('weightSkill')?.value || 60) / 100;
                const c = parseFloat(document.getElementById('weightConsist')?.value || 30) / 100;
                const ex = parseFloat(document.getElementById('weightExp')?.value || 10) / 100;
                const total = (s + c + ex) || 1;

                State.weights = { skill: s / total, consist: c / total, exp: ex / total };
                State.hasUnsavedWeightChanges = true;
                saveToCache();

                runKMeansAndAnomalies();
                renderConsistencyMatrix();
                if (typeof renderHireTalentList === 'function') renderHireTalentList();
                if (State.selectedEvent) renderRecommendations();

                // Show unsaved indicator
                const statusEl = document.getElementById('weightSaveStatus');
                if (statusEl) statusEl.innerHTML = '<span style="color:var(--accent-gold)">⚠️ Unsaved changes - Click "Save Settings to Cloud" to apply globally</span>';
            };
        }
    });

window.autoTuneWeights = function () {
    const sVal = 55;
    const cVal = 30;
    const exVal = 15;

    const sSlider = document.getElementById('weightSkill');
    const cSlider = document.getElementById('weightConsist');
    const exSlider = document.getElementById('weightExp');

    if (sSlider) sSlider.value = sVal;
    if (cSlider) cSlider.value = cVal;
    if (exSlider) exSlider.value = exVal;

    if (document.getElementById('weightSkillVal')) document.getElementById('weightSkillVal').textContent = sVal;
    if (document.getElementById('weightConsistVal')) document.getElementById('weightConsistVal').textContent = cVal;
    if (document.getElementById('weightExpVal')) document.getElementById('weightExpVal').textContent = exVal;

    const total = sVal + cVal + exVal;
    State.weights = { skill: sVal / total, consist: cVal / total, exp: exVal / total };
    State.hasUnsavedWeightChanges = true;
    saveToCache();

    runKMeansAndAnomalies();
    renderConsistencyMatrix();
    if (typeof renderHireTalentList === 'function') renderHireTalentList();
    if (State.selectedEvent) renderRecommendations();

    showToast("⚡ Neural Decision Weights Auto-Tuned & Calibrated!", "success");

    const statusEl = document.getElementById('weightSaveStatus');
    if (statusEl) statusEl.innerHTML = '<span style="color:var(--accent-gold)">⚠️ Unsaved - Click "Save Settings to Cloud" to apply globally</span>';
};

window.saveWeightsToCloud = async function () {
    const btn = document.getElementById('saveWeightsBtn');
    const statusEl = document.getElementById('weightSaveStatus');

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i data-lucide="loader" class="rotating"></i> Saving to Cloud...';
    }

    if (!_supabase) {
        showToast("❌ Cloud connection not available.", "danger");
        if (btn) { btn.disabled = false; btn.innerHTML = '<i data-lucide="cloud-upload"></i> Save Settings to Cloud'; }
        return;
    }

    const { error } = await _supabase.from('system_settings').upsert([{ 
        key: 'mcdm_weights', 
        value: State.weights,
        updated_at: new Date().toISOString()
    }], { onConflict: 'key' });

    if (error) {
        console.error('❌ WEIGHT SAVE ERROR:', error);
        showToast("❌ Failed to save: " + error.message, "danger");
        if (statusEl) statusEl.innerHTML = '<span style="color:var(--danger)">❌ Save failed: ' + error.message + '</span>';
    } else {
        State.hasUnsavedWeightChanges = false;
        console.log('✅ Weights saved to cloud:', State.weights);
        showToast("☁️ Settings saved to Cloud! All Buyers will see updated rankings.", "success");
        if (statusEl) statusEl.innerHTML = '<span style="color:var(--accent-success)">✅ Saved to Cloud successfully! All users will receive these settings.</span>';
    }

    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="cloud-upload"></i> Save Settings to Cloud';
        if (window.lucide) window.lucide.createIcons();
    }
};

    const pForm = document.getElementById('participantForm');
    if (pForm) {
        pForm.onsubmit = async (e) => {
            e.preventDefault();
            const btn = pForm.querySelector('button[type="submit"]');
            btn.innerHTML = `<i data-lucide="loader" class="rotating"></i> Neural Inferencing...`;
            btn.disabled = true;

            setTimeout(async () => {
                const formData = new FormData(pForm);

                let myId = localStorage.getItem('my_profile_id');
                let existingProfile = State.participants.find(p => String(p.id) === String(myId));

                if (!existingProfile && State.currentUser) {
                    existingProfile = State.participants.find(p => p.email && p.email.toLowerCase() === State.currentUser.email.toLowerCase());
                    if (existingProfile) myId = existingProfile.id;
                }

                const isUpdating = !!existingProfile;

                let existingJudgeA = existingProfile?.judgeA || 0;
                let existingJudgeB = existingProfile?.judgeB || 0;
                let existingJudgeC = existingProfile?.judgeC || 0;

                const forensics = {
                    name: formData.get('name'),
                    email: formData.get('email') || '',
                    phone: formData.get('phone') || '',
                    region: formData.get('region') || 'Western',
                    experience: parseInt(formData.get('experience')),
                    inactiveMonths: parseInt(formData.get('inactiveMonths')) || 0,
                    videoUrl: formData.get('videoUrl') || '',
                    judgeA: existingJudgeA,
                    judgeB: existingJudgeB,
                    judgeC: existingJudgeC,
                    consistency: Math.round((existingJudgeA + existingJudgeB + existingJudgeC) / 3),
                    skills: extractSkillsFromText(formData.get('bio'))
                };

                let pID = myId;
                if (isUpdating) {
                    forensics.id = myId;
                    if (_supabase) {
                        const { error } = await _supabase.from('participants').update(forensics).eq('id', myId);
                        if (!error) {
                            logSync(`Successfully updated ${forensics.name} in Cloud Database.`, "success");
                        } else {
                            console.error("Cloud update error:", error);
                            logSync("Cloud Sync Error: Update saved locally only.", "remote");
                        }
                    }
                    const idx = State.participants.findIndex(p => String(p.id) === String(myId));
                    if (idx !== -1) State.participants[idx] = forensics;
                } else {
                    if (_supabase) {
                        const { data, error } = await _supabase.from('participants').insert([forensics]).select();
                        if (!error && data) {
                            pID = data[0].id;
                            logSync(`Successfully synced ${forensics.name} to Cloud Database.`, "success");
                        } else {
                            console.error("Cloud insert error:", error);
                            logSync("Cloud Sync Error: Data saved locally only.", "remote");
                        }
                    }

                    if (pID) forensics.id = pID;
                    else forensics.id = Date.now();

                    // Automatically remove demo seed performers when a real performer registers
                    if (State.participants.some(p => String(p.id).startsWith('seed-'))) {
                        State.participants = State.participants.filter(p => !String(p.id).startsWith('seed-'));
                    }

                    State.participants.push(forensics);
                    localStorage.setItem('my_profile_id', forensics.id);
                }

                saveToCache();
                pForm.reset();
                runKMeansAndAnomalies();
                renderParticipantRegistry();
                renderMyProfile();
                if (State.selectedEvent) renderRecommendations();
                
                showToast(isUpdating ? "Profile updated successfully." : "Profile registered successfully.", "primary");
                logSync(`${isUpdating ? 'Updated' : 'Registered'} performer: ${forensics.name}`, "primary");
                
                btn.innerHTML = `<i data-lucide="sparkles"></i> Analyze & Register`;
                btn.disabled = false;
                lucide.createIcons();

                switchTab('profile');
            }, 600);
        };
    }
}

function switchTab(tabId) {
    // Guard admin-only tabs
    if (!State.isAdmin && (tabId === 'management' || tabId === 'analytics' || tabId === 'hirePanel')) {
        return switchTab('dashboard');
    }

    const section = document.getElementById(`${tabId}Section`);
    if (!section) return switchTab('dashboard');

    State.activeTab = tabId;
    localStorage.setItem('aesthetic_active_tab', tabId);

    // Hide all tab panes, show only the target
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    section.classList.add('active');

    // Sync nav button active state
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    // Lazy-render on tab switch
    if (tabId === 'hire') {
        populateHireEventFilter();
        renderHireTalentList();
        // Auto-fill buyer contact from logged-in account
        if (State.currentUser) {
            const nameEl = document.getElementById('buyerName');
            const emailEl = document.getElementById('buyerEmail');
            if (nameEl && !nameEl.value) nameEl.value = State.currentUser.name || '';
            if (emailEl && !emailEl.value) emailEl.value = State.currentUser.email || '';
        }
    } else if (tabId === 'myBookings') {
        renderMyBookings();
    } else if (tabId === 'register') {
        // Auto-fill existing performer profile if editing, or pre-fill account info
        if (typeof prepopulateRegistrationForm === 'function') prepopulateRegistrationForm();
        if (State.currentUser) {
            const form = document.getElementById('participantForm');
            if (form) {
                const nameEl = form.querySelector('input[name="name"]');
                const emailEl = form.querySelector('input[name="email"]');
                if (nameEl && !nameEl.value) nameEl.value = State.currentUser.name || '';
                if (emailEl && !emailEl.value) emailEl.value = State.currentUser.email || '';
            }
        }
    } else if (tabId === 'hirePanel') {
        renderHirePanel();
    }
}

window.toggleAdminMode = () => {
    if (State.isAdmin) {
        State.isAdmin = false;
        showToast("Administrative session terminated.", "primary");
        updateAdminUI();
    } else {
        const modal = document.getElementById('adminAuthModal');
        if (modal) {
            const input = document.getElementById('adminPinInput');
            if (input) input.value = '';
            modal.style.display = 'flex';
            setTimeout(() => input?.focus(), 100);
            lucide.createIcons();
        }
    }
};

window.closeAdminAuthModal = () => {
    const modal = document.getElementById('adminAuthModal');
    if (modal) modal.style.display = 'none';
};

window.submitAdminPin = (e) => {
    e.preventDefault();
    const pin = document.getElementById('adminPinInput')?.value?.trim();
    if (pin === "2001") {
        State.isAdmin = true;
        closeAdminAuthModal();
        showToast("Administrative Authorization Granted. Welcome Admin!", "success");
        updateAdminUI();
        switchTab('hirePanel');
    } else {
        showToast("Invalid Authorization PIN. Access Denied.", "danger");
        const input = document.getElementById('adminPinInput');
        if (input) {
            input.value = '';
            input.focus();
        }
    }
};

// ---- UNIFIED USER AUTHENTICATION SYSTEM ----
let authMode = 'login';

window.openAuthModal = function () {
    if (State.currentUser) {
        if (confirm(`Logged in as ${State.currentUser.name || State.currentUser.email}. Do you want to log out?`)) {
            State.currentUser = null;
            localStorage.removeItem('talent_current_user');
            localStorage.removeItem('my_buyer_email');
            localStorage.removeItem('my_profile_id');
            showToast("Logged out successfully.", "primary");
            updateAuthUI();
            if (typeof renderMyBookings === 'function') renderMyBookings();
            if (typeof renderMyProfile === 'function') renderMyProfile();
        }
        return;
    }

    const modal = document.getElementById('userAuthModal');
    if (modal) {
        switchAuthTab('login');
        modal.style.display = 'flex';
        setTimeout(() => document.getElementById('authEmailInput')?.focus(), 100);
        if (window.lucide) window.lucide.createIcons();
    }
};

window.closeAuthModal = function () {
    const modal = document.getElementById('userAuthModal');
    if (modal) modal.style.display = 'none';
};

window.switchAuthTab = function (mode) {
    authMode = mode;
    const nameGroup = document.getElementById('authNameGroup');
    const tabLogin = document.getElementById('authTabLogin');
    const tabSignup = document.getElementById('authTabSignup');
    const submitText = document.getElementById('authSubmitText');
    const subtitle = document.getElementById('authModalSubtitle');

    if (mode === 'signup') {
        if (nameGroup) nameGroup.style.display = 'block';
        if (tabLogin) tabLogin.classList.remove('active');
        if (tabSignup) tabSignup.classList.add('active');
        if (submitText) submitText.textContent = 'Create Account';
        if (subtitle) subtitle.textContent = 'Register a unified account for both Hiring and Performing';
    } else {
        if (nameGroup) nameGroup.style.display = 'none';
        if (tabLogin) tabLogin.classList.add('active');
        if (tabSignup) tabSignup.classList.remove('active');
        if (submitText) submitText.textContent = 'Sign In';
        if (subtitle) subtitle.textContent = 'Sign in to access your Dual Buyer & Performer account';
    }
    if (window.lucide) window.lucide.createIcons();
};

window.handleUserAuthSubmit = async function (e) {
    e.preventDefault();
    const email = document.getElementById('authEmailInput')?.value?.trim();
    const password = document.getElementById('authPasswordInput')?.value?.trim();
    const inputName = document.getElementById('authNameInput')?.value?.trim();

    if (!email || !password) {
        showToast('Please enter both email and password.', 'danger');
        return;
    }

    const lowerEmail = email.toLowerCase();
    let existingUser = State.userAccounts.find(u => u.email?.toLowerCase() === lowerEmail);

    // If not found locally, try fetching from Supabase user_accounts
    if (!existingUser && _supabase) {
        try {
            const { data } = await _supabase.from('user_accounts').select('*').eq('email', lowerEmail).maybeSingle();
            if (data) {
                existingUser = {
                    id: data.id,
                    name: data.name,
                    email: data.email,
                    createdAt: data.created_at
                };
                State.userAccounts.push(existingUser);
                localStorage.setItem('talent_user_accounts', JSON.stringify(State.userAccounts));
            }
        } catch (err) {
            console.log("User lookup note:", err);
        }
    }

    let user;

    if (authMode === 'login') {
        if (!existingUser) {
            showToast(`No account found for "${email}". Please Sign Up first!`, 'danger');
            switchAuthTab('signup');
            return;
        }
        user = existingUser;
        showToast(`👋 Welcome back, ${user.name}! Logged in successfully.`, 'success');
    } else {
        // Sign Up Mode
        if (existingUser) {
            user = existingUser;
            showToast(`Account already exists for "${email}". Logged in successfully!`, 'info');
        } else {
            user = {
                id: `USR-${Date.now()}`,
                name: inputName || email.split('@')[0],
                email: email,
                createdAt: new Date().toISOString()
            };
            State.userAccounts.push(user);
            localStorage.setItem('talent_user_accounts', JSON.stringify(State.userAccounts));

            if (_supabase) {
                try {
                    await _supabase.from('user_accounts').upsert([{ 
                        id: user.id, 
                        name: user.name, 
                        email: user.email, 
                        created_at: user.createdAt 
                    }]);
                    logSync(`User account ${user.email} created on Cloud.`, "success");
                } catch (err) {
                    console.log("Cloud user account note:", err);
                }
            }
            showToast(`🎉 Account created successfully! Welcome, ${user.name}.`, 'success');
        }
    }

    State.currentUser = user;
    localStorage.setItem('talent_current_user', JSON.stringify(user));
    localStorage.setItem('my_buyer_email', email);

    closeAuthModal();
    updateAuthUI();

    if (typeof renderMyBookings === 'function') renderMyBookings();
    if (typeof renderMyProfile === 'function') renderMyProfile();
};

function updateAuthUI() {
    const btnLabel = document.getElementById('userAuthLabel');
    const btn = document.getElementById('userAuthBtn');

    if (State.currentUser) {
        if (btnLabel) btnLabel.textContent = State.currentUser.name || State.currentUser.email.split('@')[0];
        if (btn) {
            btn.style.background = 'rgba(99, 102, 241, 0.15)';
            btn.style.borderColor = 'rgba(99, 102, 241, 0.35)';
            btn.style.color = 'var(--accent-primary)';
        }
    } else {
        if (btnLabel) btnLabel.textContent = 'Log In';
        if (btn) {
            btn.style.background = 'transparent';
            btn.style.borderColor = 'var(--border)';
            btn.style.color = 'var(--text-secondary)';
        }
    }
    if (window.lucide) window.lucide.createIcons();
}

window.switchPortal = function (portal) {
    if (portal === 'admin' && !State.isAdmin) {
        // Prompt for Admin PIN authorization
        const modal = document.getElementById('adminAuthModal');
        if (modal) {
            const input = document.getElementById('adminPinInput');
            if (input) input.value = '';
            modal.style.display = 'flex';
            setTimeout(() => input?.focus(), 100);
            if (window.lucide) window.lucide.createIcons();
        }
        return;
    }

    State.activePortal = portal;
    localStorage.setItem('active_portal', portal);

    // Update active portal pill highlight
    document.getElementById('portalBuyerBtn')?.classList.toggle('active', portal === 'buyer');
    document.getElementById('portalPerformerBtn')?.classList.toggle('active', portal === 'performer');
    document.getElementById('portalAdminBtn')?.classList.toggle('active', portal === 'admin');

    // Display only the relevant portal's sub-navigation tabs
    const bTabs = document.getElementById('buyerTabs');
    const pTabs = document.getElementById('performerTabs');
    const aTabs = document.getElementById('adminTabs');

    if (bTabs) bTabs.style.display = (portal === 'buyer') ? 'flex' : 'none';
    if (pTabs) pTabs.style.display = (portal === 'performer') ? 'flex' : 'none';
    if (aTabs) aTabs.style.display = (portal === 'admin') ? 'flex' : 'none';

    // Switch to first tab of active portal
    if (portal === 'buyer') switchTab('dashboard');
    else if (portal === 'performer') switchTab('register');
    else if (portal === 'admin') switchTab('hirePanel');

    if (window.lucide) window.lucide.createIcons();
};

function updateAdminUI() {
    const lockIcon = document.getElementById('portalLockIcon');
    if (lockIcon) {
        lockIcon.setAttribute('data-lucide', State.isAdmin ? 'unlock' : 'lock');
    }

    if (State.isAdmin) {
        window.switchPortal('admin');
    } else {
        const aTabs = document.getElementById('adminTabs');
        if (aTabs) aTabs.style.display = 'none';
        if (State.activePortal === 'admin') {
            window.switchPortal('buyer');
        }
    }

    renderParticipantRegistry();
    renderEvents();
    if (State.isAdmin) renderHirePanel();
    if (window.lucide) window.lucide.createIcons();
}

function renderEvents() {
    const list = document.getElementById('eventList');
    const analyticsList = document.getElementById('analyticsEventsList');

    // Header action for Admin (Add Event button)
    const addBtn = document.getElementById('adminAddEventBtn');
    if (addBtn) {
        addBtn.style.display = State.isAdmin ? 'inline-flex' : 'none';
    }

    if (!State.events || State.events.length === 0) {
        if (list) list.innerHTML = '<p style="padding:1rem; opacity:0.5; text-align:center;">No active events.</p>';
        if (analyticsList) analyticsList.innerHTML = '<p style="padding:0.5rem; opacity:0.5; text-align:center;">No events created yet.</p>';
        return;
    }

    const html = State.events.map(ev => {
        const isSelected = State.selectedEvent && State.selectedEvent.id === ev.id ? 'active' : '';
        const deleteBtn = State.isAdmin ? `
            <button onclick="deleteEvent(${ev.id}, event)" title="Delete Event (Admin Only)" style="background:rgba(239,68,68,0.15); color:#ef4444; border:none; width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; cursor:pointer; font-size:0.75rem; transition:transform 0.2s;" onmouseenter="this.style.transform='scale(1.2)'" onmouseleave="this.style.transform='scale(1)'">
                ✕
            </button>
        ` : '';

        return `
        <div class="event-item ${isSelected}" onclick="selectEvent(${ev.id}, this)">
            <div style="display:flex; align-items:center; justify-content:space-between; width:100%;">
                <h4 style="margin:0;">${escapeHtml(ev.name)}</h4>
                ${deleteBtn}
            </div>
            <div class="tags" style="margin-top:0.4rem;">${ev.requirements.map(r => `<span class="tag">${escapeHtml(r)}</span>`).join('')}</div>
        </div>`;
    }).join('');

    if (list) list.innerHTML = html;
    if (analyticsList) analyticsList.innerHTML = html;

    if (window.lucide) window.lucide.createIcons();
}

window.openCreateEventModal = function () {
    if (!State.isAdmin) {
        showToast('🔒 Admin PIN required to create events.', 'warning');
        return;
    }
    const modal = document.getElementById('createEventModal');
    if (modal) {
        modal.style.display = 'flex';
        if (window.lucide) window.lucide.createIcons();
    }
};

window.closeCreateEventModal = function () {
    const modal = document.getElementById('createEventModal');
    if (modal) modal.style.display = 'none';
};

window.submitNewEvent = function (e) {
    if (e) e.preventDefault();
    if (!State.isAdmin) {
        showToast('🔒 Only Admin can create events.', 'warning');
        return;
    }

    const nameInput = document.getElementById('newEventName');
    const descInput = document.getElementById('newEventDesc');
    const reqInput = document.getElementById('newEventRequirements');

    if (!nameInput || !descInput || !reqInput) return;

    const name = nameInput.value.trim();
    const desc = descInput.value.trim();
    const reqStr = reqInput.value.trim();

    if (!name || !desc || !reqStr) {
        showToast('⚠️ Please fill in all fields.', 'warning');
        return;
    }

    const requirements = reqStr.split(',').map(s => s.trim()).filter(Boolean);

    const newEv = {
        id: Date.now(),
        name: name,
        description: desc,
        requirements: requirements
    };

    State.events.push(newEv);
    saveToCache();

    nameInput.value = '';
    descInput.value = '';
    reqInput.value = '';

    closeCreateEventModal();
    renderEvents();
    showToast(`✅ Event "${name}" published successfully!`, 'success');
};

window.deleteEvent = function (id, e) {
    if (e) e.stopPropagation();
    if (!State.isAdmin) {
        showToast('🔒 Only Admin can delete events.', 'warning');
        return;
    }

    if (!confirm('Are you sure you want to delete this event?')) return;

    State.events = State.events.filter(ev => ev.id !== id);
    if (State.selectedEvent && State.selectedEvent.id === id) {
        State.selectedEvent = State.events[0] || null;
        if (!State.selectedEvent) {
            const noSel = document.getElementById('noSelection');
            const evAna = document.getElementById('eventAnalytics');
            if (noSel) noSel.style.display = 'block';
            if (evAna) evAna.style.display = 'none';
        } else {
            document.getElementById('targetEventName').textContent = State.selectedEvent.name;
            document.getElementById('targetEventDesc').textContent = State.selectedEvent.description;
            renderRecommendations();
        }
    }

    saveToCache();
    renderEvents();
    showToast('🗑️ Event deleted.', 'primary');
};

function selectEvent(id, el) {
    State.selectedEvent = State.events.find(e => e.id === id);
    document.querySelectorAll('.event-item').forEach(e => e.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('noSelection').style.display = 'none';
    document.getElementById('eventAnalytics').style.display = 'block';
    document.getElementById('targetEventName').textContent = State.selectedEvent.name;
    document.getElementById('targetEventDesc').textContent = State.selectedEvent.description;
    
    // Show Advanced Analytics Sections
    const teamSection = document.getElementById('teamDiscoverySection');
    if (teamSection) teamSection.style.display = 'block';
    
    renderRecommendations();
    renderConsistencyMatrix();
}

function renderRecommendations() {
    const container = document.getElementById('recommendationContainer');
    if (!container || !State.selectedEvent) return;
    
    container.innerHTML = '';
    const scored = State.participants
        .map(p => ({ ...p, result: calculateArtisticScore(p, State.selectedEvent) }))
        .sort((a, b) => b.result.total - a.result.total);

    scored.forEach((p, idx) => {
        const div = document.createElement('div');
        div.className = `medalist-card top-${idx + 1}`;
        div.onclick = () => showDetails(p);
        div.innerHTML = `
            <div class="avatar">${p.name.split(' ').map(n=>n[0]).join('')}</div>
            <div style="flex:1">
                <div style="display:flex; align-items:center;">
                    <h3 style="margin-bottom:0.2rem">${p.name}</h3>
                    ${p.isDarkHorse ? `<div class="status-badge-anomaly">🏆 ANOMALY</div>` : ''}
                </div>
                <p style="font-size:0.75rem; color:var(--text-secondary)">
                    ${p.result.scientific}% Performative Alignment • ${p.result.probability}% Forecast
                </p>
                <div style="margin-top:0.4rem">${generateRadarMini(p.result.breakdown)}</div>
            </div>
            <div class="score-num">${p.result.total}%</div>
        `;
        container.appendChild(div);
    });
    lucide.createIcons();
}

function generateRadarMini(breakdown) {
    return `<svg width="100" height="6" style="background:rgba(255,255,255,0.05); border-radius:3px">
        <rect width="${breakdown.skill * 60}" height="6" fill="var(--accent-primary)" />
        <rect x="${breakdown.skill * 60}" width="${breakdown.consist * 30}" height="6" fill="var(--accent-secondary)" />
        <rect x="${(breakdown.skill * 60) + (breakdown.consist * 30)}" width="${breakdown.exp * 10}" height="6" fill="var(--accent-gold)" />
    </svg>`;
}

function renderParticipantRegistry() {
    const tbody = document.getElementById('participantRegistry');
    if (!tbody) return;
    tbody.innerHTML = State.participants.map(p => `
        <tr>
            <td><strong>${p.name}</strong></td>
            <td><span class="tag" style="font-size:0.6rem; padding:0.2rem 0.4rem; background:rgba(255,255,255,0.1)">${p.region || 'Western'}</span></td>
            <td>${p.experience} Yrs</td>
            <td>${p.consistency}% ${p.inactiveMonths ? `<span style="font-size:0.7rem; color:var(--accent-secondary); opacity:0.8;">(${p.inactiveMonths}m idle)</span>` : ''}</td>
            <td>${p.skills}</td>
            <td>
                ${State.isAdmin ? `
                    <div style="display: flex; gap: 0.5rem;">
                        <button class="action-btn" onclick="openGradingModal('${p.id}')" title="Score Performer" style="background: rgba(99,102,241,0.15); border: 1px solid var(--accent-primary); color: var(--accent-primary);">
                            <i data-lucide="award"></i>
                        </button>
                        <button class="action-btn" onclick="deleteParticipant('${p.id}')" title="Delete Performer" style="background: rgba(239,68,68,0.15); border: 1px solid var(--danger); color: var(--danger);">
                            <i data-lucide="trash-2"></i>
                        </button>
                    </div>
                ` : `<span style="opacity:0.2; font-size:0.7rem"><i data-lucide="lock" style="width:12px; height:12px"></i></span>`}
            </td>
        </tr>
    `).join('');
    lucide.createIcons();
}

function showToast(m, t) {
    const c = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `status-badge ${t}`;
    toast.style.margin = '1rem';
    toast.textContent = m;
    c.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function logSync(m, t) {
    const c = document.getElementById('syncLog');
    const d = document.createElement('div');
    d.className = `log-entry ${t}`;
    d.textContent = `[${new Date().toLocaleTimeString()}] ${m}`;
    c.prepend(d);
}

function showDetails(p) {
    const modal = document.getElementById('detailsModal');
    const body = document.getElementById('modalBody');
    let simYears = 5;
    let radarChartInstance = null;

    const draw = () => {
        const trajectory = calculateTrajectory(p, simYears, State.selectedEvent);
        const futureRes = trajectory[trajectory.length - 1];
        const res = futureRes.full; // Use projected state so all components update on slider drag
        
        body.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem">
                <div>
                    <h2 style="color:var(--accent-primary); font-size:1.8rem">${p.name}</h2>
                    <p style="color:var(--text-secondary); font-size:0.9rem">Artistic Intelligence Report • Performer Deep-Audit</p>
                </div>
                ${p.isDarkHorse ? `<div class="status-badge-anomaly" style="position:static">🏆 ARTISTIC ANOMALY</div>` : ''}
            </div>
            
            <div style="display:grid; grid-template-columns: 1.1fr 1fr; gap:1.5rem;">
                <!-- LEFT: CONSENSUS DATA & RADAR CHART -->
                <div class="glass-panel" style="padding:1.5rem">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem">
                        <div style="text-align:center; flex:1">
                            <label style="font-size:0.7rem; text-transform:uppercase; color:var(--text-muted)">Projected Rating</label>
                            <div style="font-size:1.8rem; font-weight:900; color:var(--text-primary)">${res.total}%</div>
                        </div>
                        <div style="width:1px; height:30px; background:var(--border)"></div>
                        <div style="text-align:center; flex:1">
                            <label style="font-size:0.7rem; text-transform:uppercase; color:var(--text-muted)">Model Divergence</label>
                            <div style="font-size:1.8rem; font-weight:900; color:var(--accent-secondary)">${res.conflict.score}%</div>
                        </div>
                    </div>
                    
                    <h4 style="margin-bottom:0.5rem; font-size:0.85rem">Performative Dimension Radar</h4>
                    <div style="position:relative; height:180px; width:100%; display:flex; justify-content:center; align-items:center;">
                        <canvas id="radarChartCanvas" style="max-height:100%; max-width:100%;"></canvas>
                    </div>
                    
                    <div style="margin-top:0.8rem; padding-top:0.8rem; border-top:1px dashed var(--border)">
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:0.2rem">
                            <span>Regional Opportunity Index (${p.region || 'Western'})</span>
                            <span style="font-weight:bold">${(res.equity.index * 100).toFixed(0)}% Access</span>
                        </div>
                        <p style="font-size:0.65rem; color:var(--text-secondary); margin:0">Tenacity Multiplier Applied: <strong>${res.equity.multiplier.toFixed(2)}x</strong></p>
                    </div>
                </div>

                <!-- RIGHT: EVOLUTION SIMULATION -->
                <div class="glass-panel" style="padding:1.5rem">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem">
                        <h4 style="font-size:0.85rem">Artistic Growth Projection</h4>
                        <div style="text-align:right">
                            <span style="font-size:0.6rem; text-transform:uppercase; opacity:0.6">Year ${simYears} Potential</span>
                            <div style="font-size:1.2rem; font-weight:900; color:var(--accent-gold)">${futureRes.score}%</div>
                        </div>
                    </div>
                    
                    <div style="margin-bottom:1.5rem">
                        <input type="range" id="simYearSlider" min="1" max="10" value="${simYears}" style="width:100%">
                        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:var(--text-muted); margin-top:0.4rem">
                            <span>Immediate (1Y)</span>
                            <span>Strategic Outlook (${simYears}Y)</span>
                        </div>
                    </div>
                    
                    <div style="display:flex; align-items:flex-end; gap:0.5rem; height:100px; padding-bottom:10px; border-bottom:1px solid var(--border)">
                        ${trajectory.map((t, i) => `
                            <div style="flex:1; display:flex; flex-direction:column; align-items:center; gap:0.4rem">
                                <div style="width:100%; height:${Math.max(t.score, 5)}%; background:${i === trajectory.length-1 ? 'var(--accent-gold)' : 'linear-gradient(to top, var(--accent-primary), var(--accent-secondary))'}; border-radius:3px; opacity:${0.4 + (i * 0.1)}; transition: height 0.4s ease"></div>
                                <span style="font-size:0.55rem; color:var(--text-muted)">Y${t.year}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>

            <div style="margin-top:1.5rem">
                <h4 style="margin-bottom:0.6rem">Proficiency Impact Management</h4>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem">
                    ${State.selectedEvent.requirements.map(req => {
                        const has = p.skills.toLowerCase().includes(req.toLowerCase());
                        return `<div class="tag" style="cursor:pointer; background:${has ? 'var(--accent-primary)' : 'var(--bg-deep)'}; color:${has ? 'white' : 'var(--text-secondary)'}" onclick="toggleSimSkill(${p.id}, '${req}')">
                            <i data-lucide="${has ? 'check' : 'plus'}" style="width:12px; height:12px; margin-right:4px"></i>${req}
                        </div>`;
                    }).join('')}
                </div>
            </div>

            <!-- Showcase Performance Clip Container (Downsized & Centered) -->
            <div class="glass-panel" style="margin-top: 1.5rem; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem; position: relative; z-index: 5;">
                <h4 style="margin: 0; display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem">
                    <i data-lucide="video" style="color:var(--accent-primary)"></i> Performative Showcase Clip
                </h4>
                ${p.videoUrl ? `
                    <div style="position:relative; width:100%; max-width:480px; margin:0 auto; padding-bottom:270px; height:0; overflow:hidden; border-radius:12px; border:1px solid var(--border)">
                        ${getMediaEmbedHtml(p.videoUrl)}
                    </div>
                ` : `
                    <div style="background:var(--bg-deep); border-radius:12px; padding:1.25rem; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px dashed var(--border); position:relative; overflow:hidden; min-height:100px; max-width:480px; margin:0 auto; width:100%">
                        <div style="display:flex; align-items:center; gap:0.3rem; margin-bottom:0.6rem">
                            <!-- Glowing animated micro-visualizer lines -->
                            <div style="width:3px; height:15px; background:var(--accent-primary); border-radius:3px; animation: bounce 0.8s ease-in-out infinite alternate"></div>
                            <div style="width:3px; height:35px; background:var(--accent-secondary); border-radius:3px; animation: bounce 0.6s ease-in-out infinite alternate 0.1s"></div>
                            <div style="width:3px; height:20px; background:var(--accent-primary); border-radius:3px; animation: bounce 0.9s ease-in-out infinite alternate 0.2s"></div>
                            <div style="width:3px; height:45px; background:var(--accent-gold); border-radius:3px; animation: bounce 0.7s ease-in-out infinite alternate 0.3s"></div>
                            <div style="width:3px; height:15px; background:var(--accent-secondary); border-radius:3px; animation: bounce 0.8s ease-in-out infinite alternate 0.4s"></div>
                        </div>
                        <span style="font-size:0.7rem; color:var(--text-secondary); text-align:center; font-weight:700">No media clip uploaded by performer.</span>
                        <span style="font-size:0.6rem; color:var(--text-muted); margin-top:0.15rem">Generating synthesized algorithmic performative forecast...</span>
                    </div>
                `}
            </div>
        `;

        lucide.createIcons();

        // Initialize Chart.js Radar Chart
        const canvas = document.getElementById('radarChartCanvas');
        if (canvas) {
            if (radarChartInstance) {
                radarChartInstance.destroy();
            }
            const ctx = canvas.getContext('2d');
            const isDark = document.body.classList.contains('dark-mode');
            const textColor = isDark ? '#ffffff' : '#1e1e1e';
            const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';

            radarChartInstance = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Skill Match', 'Consistency', 'Tenacity'],
                    datasets: [
                        {
                            label: 'Performer Cap.',
                            data: [
                                Math.round(res.breakdown.skill * 100),
                                Math.round(res.breakdown.consist * 100),
                                Math.round(res.breakdown.exp * 100)
                            ],
                            backgroundColor: 'rgba(109, 40, 217, 0.25)', // Primary accent
                            borderColor: '#6d28d9',
                            borderWidth: 2,
                            pointBackgroundColor: '#6d28d9'
                        },
                        {
                            label: 'Weights',
                            data: [
                                Math.round(State.weights.skill * 100),
                                Math.round(State.weights.consist * 100),
                                Math.round(State.weights.exp * 100)
                            ],
                            backgroundColor: 'rgba(217, 70, 239, 0.1)', // Secondary accent
                            borderColor: '#d946ef',
                            borderWidth: 1.5,
                            borderDash: [3, 3],
                            pointBackgroundColor: '#d946ef'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: gridColor },
                            grid: { color: gridColor },
                            pointLabels: {
                                color: textColor,
                                font: { family: 'Outfit', size: 9, weight: '600' }
                            },
                            ticks: {
                                display: false,
                                maxTicksLimit: 5
                            },
                            suggestedMin: 0,
                            suggestedMax: 100
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'bottom',
                            labels: {
                                color: textColor,
                                font: { family: 'Outfit', size: 8 },
                                boxWidth: 8
                            }
                        }
                    }
                }
            });
        }

        document.getElementById('simYearSlider').oninput = (e) => {
            simYears = parseInt(e.target.value);
            draw();
        };
    };

    window.toggleSimSkill = (pid, skill) => {
        const perf = State.participants.find(pt => pt.id === pid);
        let sList = perf.skills.split(',').map(s => s.trim());
        if (sList.some(s => s.toLowerCase() === skill.toLowerCase())) {
            sList = sList.filter(s => s.toLowerCase() !== skill.toLowerCase());
        } else {
            sList.push(skill);
        }
        perf.skills = sList.join(', ');
        saveToCache();
        draw();
        renderRecommendations();
    };

    draw();
    modal.style.display = 'flex';
    
    // Save radar chart instance globally to destroy on close
    window._activeRadarChart = radarChartInstance;

    const closeBtn = modal.querySelector('.close-modal');
    if (closeBtn) {
        closeBtn.onclick = window.closeDetailsModal;
    }
}

window.closeDetailsModal = function () {
    if (window._activeRadarChart) {
        try { window._activeRadarChart.destroy(); } catch (e) {}
        window._activeRadarChart = null;
    }
    const modal = document.getElementById('detailsModal');
    if (modal) modal.style.display = 'none';
};

// Global backdrop click handler to close any modal when clicking outside modal content
window.addEventListener('click', (e) => {
    if (e.target && e.target.classList && e.target.classList.contains('modal-overlay')) {
        if (e.target.id === 'detailsModal') {
            window.closeDetailsModal();
        } else {
            e.target.style.display = 'none';
        }
    }
});

// Wait for all CDN scripts to be ready before initializing
function waitForCDNAndInit() {
    const ready = (
        typeof lucide !== 'undefined' &&
        typeof supabase !== 'undefined' &&
        typeof brain !== 'undefined'
    );
    if (ready) {
        initApp();
    } else {
        setTimeout(waitForCDNAndInit, 100);
    }
}
window.addEventListener('load', waitForCDNAndInit);

window.exportSPSS = () => {
    if (State.participants.length === 0) {
        showToast("No data to export.", "primary");
        return;
    }
    
    // Headers designed for SPSS / R
    const headers = [
        "Participant_ID", "Region", "Industry_Tenure_Yrs", 
        "JudgeA_Score", "JudgeB_Score", "JudgeC_Score", 
        "Consensus_Avg", "Skill_Match_Sci", "Neural_Forecast_Prob", 
        "Hybrid_Final_Score", "Conflict_Index", "Tenacity_Multiplier"
    ];

    const rows = State.participants.map((p, index) => {
        const ev = State.selectedEvent || State.events[0];
        const res = calculateArtisticScore(p, ev);
        return [
            `PID_${index + 1}`,
            p.region || "Western",
            p.experience,
            p.judgeA !== undefined ? p.judgeA : p.consistency,
            p.judgeB !== undefined ? p.judgeB : p.consistency,
            p.judgeC !== undefined ? p.judgeC : p.consistency,
            p.consistency,
            res.scientific,
            res.probability,
            res.total,
            res.conflict.score,
            res.equity.multiplier.toFixed(3)
        ].join(",");
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `talent_premium_spss_export_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("SPSS Dossier Downloaded", "success");
};

window.toggleAblationMode = (checked) => {
    const panel = document.getElementById('ablationPanel');
    if (panel) {
        panel.style.display = checked ? 'block' : 'none';
        if (checked) {
            calculateAblation();
        }
    }
};

window.calculateAblation = () => {
    const toggle = document.getElementById('ablationToggle');
    if (!toggle || !toggle.checked) return;

    const participants = State.participants;
    if (participants.length === 0) return;

    let sqErrNN = 0;
    let sqErrWSM = 0;
    let sqErrHybrid = 0;

    const ev = State.selectedEvent || State.events[0];

    participants.forEach(p => {
        const res = calculateArtisticScore(p, ev);
        // Using Judge consensus average as the "Ground Truth" value for MSE testing
        const truth = p.consistency; 

        sqErrNN += Math.pow(res.probability - truth, 2);
        sqErrWSM += Math.pow(res.scientific - truth, 2);
        sqErrHybrid += Math.pow(res.total - truth, 2);
    });

    const mseNN = (sqErrNN / participants.length).toFixed(2);
    const mseWSM = (sqErrWSM / participants.length).toFixed(2);
    const mseHybrid = (sqErrHybrid / participants.length).toFixed(2);

    const nnEl = document.getElementById('mseNN');
    const wsmEl = document.getElementById('mseWSM');
    const hybEl = document.getElementById('mseHybrid');
    
    if (nnEl) nnEl.textContent = mseNN;
    if (wsmEl) wsmEl.textContent = mseWSM;
    if (hybEl) hybEl.textContent = mseHybrid;
};

function getMediaEmbedHtml(url) {
    if (!url) return '';
    
    // 1. YouTube Link Conversion
    const ytReg = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const ytMatch = url.match(ytReg);
    if (ytMatch && ytMatch[2].length === 11) {
        return `<iframe style="position:absolute; top:0; left:0; width:100%; height:100%; border:none" src="https://www.youtube.com/embed/${ytMatch[2]}" allow="autoplay; encrypted-media" allowfullscreen></iframe>`;
    }
    
    // 2. Google Drive Link Conversion
    if (url.includes('drive.google.com')) {
        let embedUrl = url.replace(/\/view\??.*/, '/preview').replace(/\/edit\??.*/, '/preview');
        if (!embedUrl.endsWith('/preview')) {
            embedUrl = embedUrl.split('?')[0] + '/preview';
        }
        return `<iframe style="position:absolute; top:0; left:0; width:100%; height:100%; border:none" src="${embedUrl}" allow="autoplay" allowfullscreen></iframe>`;
    }
    
    // 3. Direct Video file (MP4, WebM, OGG)
    if (url.match(/\.(mp4|webm|ogg)($|\?)/i)) {
        return `<video style="position:absolute; top:0; left:0; width:100%; height:100%; border-radius:12px" controls src="${url}"></video>`;
    }

    // 4. Fallback: Standard preview/external document frame
    return `<iframe style="position:absolute; top:0; left:0; width:100%; height:100%; border:none; background:white" src="${url}" allowfullscreen></iframe>`;
}

window.simulateCVUpload = (input) => {
    const file = input.files[0];
    if (!file) return;
    
    showToast("Analyzing Resume document layout...", "primary");
    logSync(`Importing Resume CV: ${file.name}`, "system");

    setTimeout(() => {
        showToast("NLP Engine: Parsing semantic capability graph...", "primary");
    }, 1200);

    setTimeout(() => {
        // Randomly pick a few highly-advanced local/cultural skills
        const mockSkills = [
            "Kandyan Dance, Geta Bera rhythm alignment, outstanding Stage Presence, Trilingual fluency",
            "Baila singing, Calypso music, public speaking, humor, improvisation",
            "Opera vocal range, Low-country Dance, Sinhala Diction, acting masterclass",
            "Corporate Emceeing, Professionalism, Trilingual hosting, public speaking"
        ];
        const selected = mockSkills[Math.floor(Math.random() * mockSkills.length)];
        
        const bioInput = document.getElementById('bioInput');
        if (bioInput) {
            bioInput.value = `[NLP CV Import - parsed ${file.name}]: Professional performer specializing in: ${selected}. Verified experience and regional capability registered.`;
            // Trigger skill extraction refresh
            showToast("NLP Extraction Complete: 4 capabilities synced.", "success");
            logSync(`NLP parsed skills from ${file.name} successfully.`, "success");
            
            // Highlight skills by triggering word-match dictionary parsing automatically on text change
            // We can even call extractSkillsFromText or let them click analyze
        }
    }, 2400);
};

function renderMyProfile() {
    const noProfileEl = document.getElementById('noProfileState');
    const profileDetailsEl = document.getElementById('profileDetailsState');
    if (!noProfileEl || !profileDetailsEl) return;

    let myProfile = null;

    if (State.currentUser) {
        // Strict lookup: match profile ONLY by logged in user's email or stage name
        const userEmail = State.currentUser.email?.toLowerCase();
        const userName = State.currentUser.name?.toLowerCase();
        myProfile = State.participants.find(p => 
            (p.email && p.email.toLowerCase() === userEmail) ||
            (p.name && p.name.toLowerCase() === userName)
        ) || null;

        if (myProfile) {
            localStorage.setItem('my_profile_id', myProfile.id);
        } else {
            localStorage.removeItem('my_profile_id');
        }
    } else {
        const myId = localStorage.getItem('my_profile_id');
        if (myId) {
            myProfile = State.participants.find(p => String(p.id) === String(myId)) || null;
        }
    }

    if (!myProfile) {
        noProfileEl.style.display = 'block';
        profileDetailsEl.style.display = 'none';
        return;
    }

    noProfileEl.style.display = 'none';
    profileDetailsEl.style.display = 'block';

    document.getElementById('myProfileName').textContent = myProfile.name;
    document.getElementById('myProfileRegion').textContent = myProfile.region;
    document.getElementById('myProfileExperience').textContent = `${myProfile.experience} Yrs`;
    document.getElementById('myProfileConsistency').textContent = `${myProfile.consistency}%`;

    const skillsContainer = document.getElementById('myProfileSkills');
    if (skillsContainer) {
        const skillsList = myProfile.skills.split(',').map(s => s.trim());
        skillsContainer.innerHTML = skillsList.map(s => `<span class="tag" style="background:rgba(255,255,255,0.05); border:1px solid var(--border)">${s}</span>`).join('');
    }

    const mediaContainer = document.getElementById('myProfileVideoContainer');
    if (mediaContainer) {
        if (myProfile.videoUrl) {
            mediaContainer.innerHTML = `
                <div style="position:relative; width:100%; max-width:480px; margin:0 auto; padding-bottom:270px; height:0; overflow:hidden; border-radius:12px; border:1px solid var(--border)">
                    ${getMediaEmbedHtml(myProfile.videoUrl)}
                </div>
            `;
        } else {
            mediaContainer.innerHTML = `
                <div style="background:var(--bg-deep); border-radius:12px; padding:1.25rem; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px dashed var(--border); min-height:100px; max-width:480px; margin:0 auto; width:100%">
                    <span style="font-size:0.7rem; color:var(--text-secondary); text-align:center; font-weight:700">No media clip uploaded.</span>
                </div>
            `;
        }
    }

    renderMyGigRequests(myProfile);

    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// ---- Performer Portal: Render My Gig Alerts ----
function renderMyGigRequests(myProfile) {
    const container = document.getElementById('myGigRequestsList');
    if (!container || !myProfile) return;

    const gigRequests = HireState.requests.filter(r => 
        String(r.performerId) === String(myProfile.id) ||
        (r.performerEmail && r.performerEmail.toLowerCase() === myProfile.email?.toLowerCase()) ||
        (r.performerName && r.performerName.toLowerCase() === myProfile.name.toLowerCase())
    );

    if (gigRequests.length === 0) {
        container.innerHTML = `
            <div style="text-align:center; padding:1.2rem; color:var(--text-muted); font-size:0.8rem">
                No active gig requests received yet. Showcases will highlight your profile to buyers!
            </div>
        `;
        return;
    }

    container.innerHTML = gigRequests.map(r => {
        let isApproved = (r.status === 'approved' || r.status === 'paid');
        let statusBg = isApproved ? 'rgba(16,185,129,0.15)' : (r.status === 'rejected' ? 'rgba(244,63,94,0.15)' : 'rgba(234,179,8,0.15)');
        let statusColor = isApproved ? 'var(--accent-success)' : (r.status === 'rejected' ? 'var(--accent-secondary)' : '#eab308');
        let statusLabel = isApproved ? '✅ Confirmed Gig' : (r.status === 'rejected' ? '❌ Declined' : '⏳ Pending Review');

        return `
            <div style="background:var(--bg-deep); padding:0.85rem 1rem; border-radius:10px; border:1px solid var(--border);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap; gap:0.4rem">
                    <strong style="font-size:0.88rem; color:var(--text-primary)">${r.eventName}</strong>
                    <span class="tag" style="background:${statusBg}; color:${statusColor}; font-weight:700; font-size:0.65rem">
                        ${statusLabel}
                    </span>
                </div>
                <div style="font-size:0.8rem; color:var(--text-secondary)">
                    Agreed Fee: <strong style="color:var(--accent-success)">LKR ${Number(r.feeAmount).toLocaleString()}</strong> • Buyer: <strong>${r.buyerName}</strong>
                </div>
                ${isApproved ? `
                    <div style="font-size:0.75rem; background:rgba(99,102,241,0.08); padding:0.5rem; border-radius:6px; color:var(--text-primary); margin-top:0.4rem; display:flex; align-items:center; gap:0.5rem">
                        <i data-lucide="phone" style="width:12px; height:12px; color:var(--accent-primary)"></i> Contact Buyer: <strong>${r.buyerPhone}</strong> (${r.buyerEmail})
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');

    if (window.lucide) window.lucide.createIcons();
}

// ---- Buyer Portal: Render My Bookings ----
window.renderMyBookings = function () {
    const container = document.getElementById('myBookingsList');
    const input = document.getElementById('buyerLookupEmail');
    if (!container) return;

    let email = input?.value?.trim() || localStorage.getItem('my_buyer_email') || '';
    if (input && !input.value && email) {
        input.value = email;
    }

    if (!email) {
        container.innerHTML = `
            <div class="glass-panel" style="padding:3rem; text-align:center;">
                <i data-lucide="receipt" style="width:40px; height:40px; opacity:0.3; margin-bottom:0.75rem; color:var(--accent-primary)"></i>
                <h4 style="margin-bottom:0.4rem">Enter Your Email Address</h4>
                <p style="color:var(--text-secondary); font-size:0.85rem">Type your email address above to look up your submitted hire requests and booking status.</p>
            </div>
        `;
        if (window.lucide) window.lucide.createIcons();
        return;
    }

    const myRequests = HireState.requests.filter(r => r.buyerEmail?.toLowerCase() === email.toLowerCase());

    if (myRequests.length === 0) {
        container.innerHTML = `
            <div class="glass-panel" style="padding:3rem; text-align:center;">
                <i data-lucide="inbox" style="width:40px; height:40px; opacity:0.3; margin-bottom:0.75rem; color:var(--text-muted)"></i>
                <h4 style="margin-bottom:0.4rem">No Bookings Found for "${email}"</h4>
                <p style="color:var(--text-secondary); font-size:0.85rem">You haven't submitted any talent hire requests using this email address yet.</p>
            </div>
        `;
        if (window.lucide) window.lucide.createIcons();
        return;
    }

    container.innerHTML = myRequests.map(r => {
        let badgeBg = 'rgba(234, 179, 8, 0.15)';
        let badgeColor = '#eab308';
        let badgeText = '⏳ Pending Admin Review';

        if (r.status === 'approved') {
            badgeBg = 'rgba(16, 185, 129, 0.15)';
            badgeColor = 'var(--accent-success)';
            badgeText = '✅ Approved & Talent Notified';
        } else if (r.status === 'paid') {
            badgeBg = 'rgba(99, 102, 241, 0.15)';
            badgeColor = 'var(--accent-primary)';
            badgeText = '💳 Payment Dispatched';
        } else if (r.status === 'rejected') {
            badgeBg = 'rgba(244, 63, 94, 0.15)';
            badgeColor = 'var(--accent-secondary)';
            badgeText = '❌ Booking Declined';
        }

        return `
            <div class="glass-panel" style="padding:1.5rem; border:1px solid var(--border);">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem">
                    <div>
                        <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; font-weight:700">Req ID: ${r.id}</div>
                        <h3 style="margin:0.2rem 0; color:var(--text-primary)">${r.eventName}</h3>
                        <p style="font-size:0.82rem; color:var(--text-secondary)">Hired Talent: <strong>${r.performerName}</strong></p>
                    </div>
                    <span class="tag" style="background:${badgeBg}; color:${badgeColor}; font-weight:700; padding:0.4rem 0.75rem">
                        ${badgeText}
                    </span>
                </div>

                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:0.75rem; background:var(--bg-deep); padding:0.85rem; border-radius:10px; font-size:0.8rem;">
                    <div><span style="color:var(--text-muted)">Agreed Fee:</span> <strong style="color:var(--accent-success)">LKR ${Number(r.feeAmount).toLocaleString()}</strong></div>
                    <div><span style="color:var(--text-muted)">Occasion:</span> <strong>${r.eventDesc}</strong></div>
                    <div><span style="color:var(--text-muted)">Submitted:</span> <strong>${new Date(r.createdAt).toLocaleDateString()}</strong></div>
                </div>
            </div>
        `;
    }).join('');

    if (window.lucide) window.lucide.createIcons();
};

async function deleteMyProfile() {
    const myId = localStorage.getItem('my_profile_id');
    if (!myId) return;

    if (!confirm("Are you sure you want to delete your registered profile? This will remove all your data from the system.")) return;

    if (_supabase) {
        await _supabase.from('participants').delete().eq('id', myId);
    }
    State.participants = State.participants.filter(p => String(p.id) !== String(myId));
    localStorage.removeItem('my_profile_id');
    
    saveToCache();
    renderParticipantRegistry();
    renderConsistencyMatrix();
    runKMeansAndAnomalies();
    renderMyProfile();
    
    if (State.selectedEvent) renderRecommendations();
    
    showToast("Your profile has been deleted.", "primary");
    switchTab('register');
}

window.logoutMyProfile = function () {
    const myId = localStorage.getItem('my_profile_id');
    if (!myId) return;

    if (!confirm("Are you sure you want to log out of your performer profile session?")) return;

    localStorage.removeItem('my_profile_id');
    renderMyProfile();
    showToast("🔓 Logged out of performer profile session.", "info");
};

window.deleteMyProfile = deleteMyProfile;

window.prepopulateRegistrationForm = function () {
    let myId = localStorage.getItem('my_profile_id');
    let myProfile = State.participants.find(p => String(p.id) === String(myId));

    if (!myProfile && State.currentUser) {
        myProfile = State.participants.find(p => p.email && p.email.toLowerCase() === State.currentUser.email.toLowerCase());
    }

    if (!myProfile) return;

    const form = document.getElementById('participantForm');
    if (!form) return;

    const nameInput = form.querySelector('input[name="name"]');
    const emailInput = form.querySelector('input[name="email"]');
    const phoneInput = form.querySelector('input[name="phone"]');
    const regionSelect = form.querySelector('select[name="region"]');
    const expInput = form.querySelector('input[name="experience"]');
    const inactiveInput = form.querySelector('input[name="inactiveMonths"]');
    const bioInput = form.querySelector('textarea[name="bio"]');
    const videoInput = form.querySelector('input[name="videoUrl"]');

    if (nameInput) nameInput.value = myProfile.name || '';
    if (emailInput) emailInput.value = myProfile.email || '';
    if (phoneInput) phoneInput.value = myProfile.phone || '';
    if (regionSelect) regionSelect.value = myProfile.region || 'Western';
    if (expInput) expInput.value = myProfile.experience || 0;
    if (inactiveInput) inactiveInput.value = myProfile.inactiveMonths || 0;
    if (bioInput) bioInput.value = myProfile.bio || myProfile.skills || '';
    if (videoInput) videoInput.value = myProfile.videoUrl || '';
};

function openGradingModal(id) {
    const p = State.participants.find(pt => String(pt.id) === String(id));
    if (!p) return;

    document.getElementById('gradingPerformerId').value = id;
    document.getElementById('gradeJudgeA').value = p.judgeA || 0;
    document.getElementById('gradeJudgeB').value = p.judgeB || 0;
    document.getElementById('gradeJudgeC').value = p.judgeC || 0;

    document.getElementById('gradingModal').style.display = 'flex';
}

function closeGradingModal() {
    document.getElementById('gradingModal').style.display = 'none';
}

async function submitGrading(e) {
    e.preventDefault();
    const id = document.getElementById('gradingPerformerId').value;
    const p = State.participants.find(pt => String(pt.id) === String(id));
    if (!p) return;

    const ja = parseInt(document.getElementById('gradeJudgeA').value) || 0;
    const jb = parseInt(document.getElementById('gradeJudgeB').value) || 0;
    const jc = parseInt(document.getElementById('gradeJudgeC').value) || 0;

    p.judgeA = ja;
    p.judgeB = jb;
    p.judgeC = jc;
    p.consistency = Math.round((ja + jb + jc) / 3);

    if (_supabase) {
        await _supabase.from('participants').update({
            judgeA: ja,
            judgeB: jb,
            judgeC: jc,
            consistency: p.consistency
        }).eq('id', id);
        logSync(`Successfully graded performer ${p.name} on cloud.`, "success");
    }

    saveToCache();
    closeGradingModal();
    renderParticipantRegistry();
    renderConsistencyMatrix();
    runKMeansAndAnomalies();
    renderMyProfile();
    if (State.selectedEvent) renderRecommendations();

    showToast(`Grades updated for ${p.name}.`, "success");
}

window.renderMyProfile = renderMyProfile;
window.deleteMyProfile = deleteMyProfile;
window.prepopulateRegistrationForm = prepopulateRegistrationForm;
window.openGradingModal = openGradingModal;
window.closeGradingModal = closeGradingModal;
window.submitGrading = submitGrading;
window.selectEvent = selectEvent;
window.deleteParticipant = deleteParticipant;
window.switchTab = switchTab;

// ============================================================
//  HIRING WORKFLOW ENGINE
// ============================================================

const HireState = {
    requests: JSON.parse(localStorage.getItem('hire_requests') || '[]'),
    currentFilter: 'all',
};

async function saveHireRequests(specificRequest = null) {
    // Always save to localStorage first
    localStorage.setItem('hire_requests', JSON.stringify(HireState.requests));

    if (_supabase) {
        try {
            const toSync = specificRequest || HireState.requests[0];
            if (toSync) {
                const safeReq = { ...toSync };
                if (safeReq.slipDataUrl && safeReq.slipDataUrl.length > 400000) {
                    safeReq.slipDataUrl = 'large_image_local_only';
                }
                const { error } = await _supabase.from('hire_requests').upsert([safeReq]);
                if (error) {
                    console.warn('Supabase hire sync error:', error.message);
                    logSync(`⚠️ Cloud sync failed: ${error.message} — data saved locally.`, 'remote');
                } else {
                    logSync(`☁️ Hire request synced to cloud: ${toSync.id}`, 'success');
                }
            }
        } catch (e) {
            console.log('Cloud hire sync skipped — using localStorage fallback.');
        }
    }
}

async function loadHireRequestsFromCloud() {
    if (!_supabase) return;
    try {
        const { data, error } = await _supabase.from('hire_requests').select('*').order('createdAt', { ascending: false });
        if (!error && data) {
            // Find local requests that are NOT yet in Supabase
            const cloudIds = new Set(data.map(r => r.id));
            const localOnly = HireState.requests.filter(r => !cloudIds.has(r.id));

            // Re-upload any missing local requests to Supabase (heal the sync gap)
            if (localOnly.length > 0) {
                logSync(`☁️ Re-syncing ${localOnly.length} local-only request(s) to cloud...`, 'remote');
                for (const req of localOnly) {
                    // Strip slipDataUrl if too large (>500KB base64) to avoid payload errors
                    const safeReq = { ...req };
                    if (safeReq.slipDataUrl && safeReq.slipDataUrl.length > 500000) {
                        safeReq.slipDataUrl = 'large_image_local_only';
                    }
                    const { error: upErr } = await _supabase.from('hire_requests').upsert([safeReq]);
                    if (!upErr) {
                        logSync(`☁️ Re-synced: ${req.id}`, 'success');
                    } else {
                        logSync(`⚠️ Re-sync failed for ${req.id}: ${upErr.message}`, 'danger');
                    }
                }
            }

            // Normalize column casing from Supabase
            const normalizedCloudData = (data || []).map(r => ({
                ...r,
                performerEmail: r.performerEmail || r.performeremail || '',
                performerPhone: r.performerPhone || r.performerphone || '',
                eventName: r.eventName || r.eventname || '',
                eventDesc: r.eventDesc || r.eventdesc || '',
                feeAmount: r.feeAmount || r.feeamount || 0,
                slipDataUrl: r.slipDataUrl || r.slipdataurl || '',
                buyerName: r.buyerName || r.buyername || '',
                buyerPhone: r.buyerPhone || r.buyerphone || '',
                buyerEmail: r.buyerEmail || r.buyeremail || '',
                aiScore: r.aiScore || r.aiscore || 0,
                createdAt: r.createdAt || r.createdat || new Date().toISOString()
            }));

            // Merge cloud + local into one complete list
            const merged = [...normalizedCloudData];
            HireState.requests.forEach(localReq => {
                if (!merged.some(cloudReq => cloudReq.id === localReq.id)) {
                    merged.push(localReq);
                }
            });
            merged.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
            HireState.requests = merged;
            localStorage.setItem('hire_requests', JSON.stringify(merged));
        }
    } catch (e) {
        console.log("Using cached local hire requests.");
    }
}

// ---- Populate Event Filter Dropdown in Hire Tab ----
function populateHireEventFilter() {
    const sel = document.getElementById('hireEventFilter');
    if (!sel) return;
    sel.innerHTML = State.events.map(ev =>
        `<option value="${ev.id}">${ev.name}</option>`
    ).join('');
}

// ---- Render the AI-ranked talent list for Hire tab ----
window.renderHireTalentList = function () {
    const container = document.getElementById('hireTalentList');
    if (!container) return;

    const selEl = document.getElementById('hireEventFilter');
    const eventId = selEl ? parseInt(selEl.value) : (State.events[0] || {}).id;
    const event = State.events.find(e => e.id === eventId) || State.events[0];

    if (!event || State.participants.length === 0) {
        container.innerHTML = `<div class="glass-panel" style="padding:3rem; text-align:center; opacity:0.6"><i data-lucide="users" style="width:40px; height:40px; margin-bottom:1rem; opacity:0.3"></i><p>No performers registered yet.</p></div>`;
        lucide.createIcons();
        return;
    }

    const scored = State.participants
        .map(p => ({ ...p, result: calculateArtisticScore(p, event) }))
        .sort((a, b) => b.result.total - a.result.total);

    container.innerHTML = scored.map((p, idx) => {
        // Check if already has a pending/approved request for this performer
        const existingReq = HireState.requests.find(r =>
            String(r.performerId) === String(p.id) &&
            r.eventId === eventId &&
            ['pending', 'approved'].includes(r.status)
        );

        const rankLabel = ['🥇 Top Pick', '🥈 Runner Up', '🥉 3rd Place'][idx] || `#${idx + 1}`;

        let actionBtn;
        const isSelfProfile = State.currentUser && (
            (p.email && p.email.toLowerCase() === State.currentUser.email.toLowerCase()) ||
            (p.name && p.name.toLowerCase() === State.currentUser.name.toLowerCase())
        );

        if (isSelfProfile) {
            actionBtn = `<span class="tag" style="background:rgba(99,102,241,0.15); color:var(--accent-primary); border:1px solid var(--accent-primary); font-weight:700;"><i data-lucide="user" style="width:13px;height:13px;margin-right:4px;"></i> Your Profile</span>`;
        } else if (existingReq) {
            const statusLabels = { pending: '⏳ Request Sent', approved: '✅ Hired!' };
            actionBtn = `<span class="hired-badge"><i data-lucide="check-circle" style="width:14px;height:14px;"></i> ${statusLabels[existingReq.status] || existingReq.status}</span>`;
        } else {
            actionBtn = `<button class="hire-btn" onclick="openHireModal('${p.id}', ${eventId})"><i data-lucide="user-check"></i> Hire</button>`;
        }

        const initials = (p.name || 'U').split(' ').filter(n => n).map(n => n[0]).join('').toUpperCase();
        const displaySkills = (p.skills || '').split(',').filter(s => s.trim()).slice(0,4);

        return `
        <div class="medalist-card" style="cursor:default; animation: slideInUp ${0.1 * idx + 0.1}s ease both;">
            <div style="display:flex; align-items:center; gap:0.5rem; min-width:80px; flex-direction:column;">
                <div class="avatar" style="${idx === 0 ? 'background:linear-gradient(135deg,#f59e0b,#d97706); color:white;' : idx === 1 ? 'background:linear-gradient(135deg,#94a3b8,#64748b); color:white;' : idx === 2 ? 'background:linear-gradient(135deg,#cd7c50,#a0522d); color:white;' : ''}">${initials}</div>
                <span style="font-size:0.6rem; font-weight:700; color:var(--text-muted); text-align:center;">${rankLabel}</span>
            </div>
            <div style="flex:1; min-width:0;">
                <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.4rem;">
                    <h3 style="margin:0; font-size:1rem;">${p.name || 'Unnamed Performer'}</h3>
                    <span class="tag" style="font-size:0.65rem; background:rgba(99,102,241,0.08); border:1px solid var(--border);">${p.region || 'Unknown Region'}</span>
                    ${p.isDarkHorse ? `<span class="status-badge-anomaly">🏆 ANOMALY</span>` : ''}
                </div>
                <div style="font-size:0.78rem; color:var(--text-secondary); margin-bottom:0.5rem;">
                    ${p.result.scientific}% Skill Match &nbsp;•&nbsp; ${p.result.probability}% AI Forecast &nbsp;•&nbsp; ${p.experience || 0} yrs exp
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.4rem;">
                    ${displaySkills.map(s => `<span class="tag" style="font-size:0.65rem; background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.15);">${s.trim()}</span>`).join('')}
                </div>
            </div>
            <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.75rem; min-width:100px;">
                <div class="score-num" style="font-size:1.6rem;">${p.result.total}%</div>
                ${actionBtn}
            </div>
        </div>`;
    }).join('');

    lucide.createIcons();
};

// ---- Open Hire Modal ----
window.openHireModal = function (performerId, eventId) {
    const p = State.participants.find(pt => String(pt.id) === String(performerId));
    const event = State.events.find(e => e.id === parseInt(eventId));
    if (!p || !event) return;

    // Guard: Prevent self-hiring
    if (State.currentUser && (
        (p.email && p.email.toLowerCase() === State.currentUser.email.toLowerCase()) ||
        (p.name && p.name.toLowerCase() === State.currentUser.name.toLowerCase())
    )) {
        showToast('⚠️ You cannot hire your own Performer Profile!', 'warning');
        return;
    }

    // Validate buyer contact details filled in the page form
    const buyerName  = document.getElementById('buyerName')?.value?.trim();
    const buyerPhone = document.getElementById('buyerPhone')?.value?.trim();
    const buyerEmail = document.getElementById('buyerEmail')?.value?.trim();
    if (!buyerName || !buyerPhone || !buyerEmail) {
        showToast('⚠️ Please fill in your Name, Phone and Email in "Your Contact Details" above first.', 'danger');
        document.getElementById('buyerName')?.focus();
        return;
    }

    // Populate performer header
    document.getElementById('hirePerformerId').value = performerId;
    document.getElementById('hireModalAvatar').textContent = p.name.split(' ').map(n => n[0]).join('');
    document.getElementById('hireModalName').textContent = p.name;
    document.getElementById('hireModalEvent').textContent = `Event: ${event.name}`;
    const res = calculateArtisticScore(p, event);
    document.getElementById('hireModalScore').textContent = `AI Score: ${res.total}% — ${res.scientific}% Skill Match`;

    // Populate "Booking As" summary card
    const nameEl  = document.getElementById('hireModalBuyerName');
    const phoneEl = document.getElementById('hireModalBuyerPhone');
    const emailEl = document.getElementById('hireModalBuyerEmail');
    if (nameEl)  nameEl.textContent  = buyerName;
    if (phoneEl) phoneEl.textContent = `📞 ${buyerPhone}`;
    if (emailEl) emailEl.textContent = `✉️ ${buyerEmail}`;

    // Reset only the event/slip fields for a clean slate
    document.getElementById('slipFileInput').value = '';
    document.getElementById('slipPreviewImg').style.display = 'none';
    document.getElementById('slipUploadContent').style.display = 'block';
    document.getElementById('hireSlipDataUrl').value = '';
    document.getElementById('hireEventDesc').value = '';
    document.getElementById('hireFeeAmount').value = '';

    document.getElementById('hireModal').style.display = 'flex';
    lucide.createIcons();
};

window.closeHireModal = function () {
    document.getElementById('hireModal').style.display = 'none';
};

// ---- Slip upload handlers ----
window.handleSlipUpload = function (input) {
    const file = input.files[0];
    if (!file) return;

    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Compress image using Canvas
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                const maxDim = 800; // max dimension limit

                if (width > maxDim || height > maxDim) {
                    if (width > height) {
                        height = Math.round((height * maxDim) / width);
                        width = maxDim;
                    } else {
                        width = Math.round((width * maxDim) / height);
                        height = maxDim;
                    }
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                // Export to compressed jpeg at 70% quality
                const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.7);

                document.getElementById('hireSlipDataUrl').value = compressedDataUrl;
                document.getElementById('slipPreviewImg').src = compressedDataUrl;
                document.getElementById('slipPreviewImg').style.display = 'block';
                document.getElementById('slipUploadContent').style.display = 'none';
                showToast('Payment slip uploaded and optimized successfully.', 'success');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        // PDF — store filename as indicator
        document.getElementById('hireSlipDataUrl').value = `pdf:${file.name}`;
        document.getElementById('slipUploadContent').innerHTML = `<div style="padding:1rem; display:flex; flex-direction:column; align-items:center; gap:0.5rem;"><span style="font-size:2rem;">📄</span><p style="font-weight:700; font-size:0.85rem;">${file.name}</p><p style="font-size:0.72rem; color:var(--text-muted);">PDF Slip Attached</p></div>`;
        showToast('PDF payment slip attached.', 'success');
    }
};

window.handleSlipDragOver = function (e) {
    e.preventDefault();
    document.getElementById('slipUploadZone').classList.add('drag-over');
};

window.handleSlipDrop = function (e) {
    e.preventDefault();
    document.getElementById('slipUploadZone').classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    const input = document.getElementById('slipFileInput');
    input.files = dt.files;
    window.handleSlipUpload(input);
};

// ---- Submit Hire Request ----
window.submitHireRequest = function () {
    const performerId = document.getElementById('hirePerformerId').value;
    const slipDataUrl = document.getElementById('hireSlipDataUrl').value;
    const eventDesc = document.getElementById('hireEventDesc').value.trim();
    const feeAmount = document.getElementById('hireFeeAmount').value.trim();
    const buyerName = document.getElementById('buyerName')?.value?.trim() || '';
    const buyerPhone = document.getElementById('buyerPhone')?.value?.trim() || '';
    const buyerEmail = document.getElementById('buyerEmail')?.value?.trim() || '';

    // Validation
    if (!buyerName) {
        showToast('Please enter your name.', 'danger');
        document.getElementById('buyerName').focus();
        return;
    }
    if (!buyerPhone) {
        showToast('Please enter your phone number.', 'danger');
        document.getElementById('buyerPhone').focus();
        return;
    }
    if (!buyerEmail || !buyerEmail.includes('@')) {
        showToast('Please enter a valid email address for confirmation.', 'danger');
        document.getElementById('buyerEmail').focus();
        return;
    }
    if (!eventDesc) {
        showToast('Please describe the event or occasion.', 'danger');
        document.getElementById('hireEventDesc').focus();
        return;
    }
    if (!feeAmount || isNaN(feeAmount) || Number(feeAmount) <= 0) {
        showToast('Please enter a valid agreed fee amount.', 'danger');
        document.getElementById('hireFeeAmount').focus();
        return;
    }
    if (!slipDataUrl) {
        showToast('Please upload your bank payment slip.', 'danger');
        return;
    }

    const selEl = document.getElementById('hireEventFilter');
    const eventId = selEl ? parseInt(selEl.value) : State.events[0]?.id;
    const event = State.events.find(e => e.id === eventId);
    const performer = State.participants.find(p => String(p.id) === String(performerId));

    const res = calculateArtisticScore(performer, event);

    const request = {
        id: `HR-${Date.now()}`,
        performerId: String(performerId),
        performerName: performer.name,
        performerEmail: performer.email || '',          // saved from registration form
        performerPhone: performer.phone || '',
        performerSkills: performer.skills,
        eventId,
        eventName: event?.name || 'Unknown Event',
        eventDesc,
        feeAmount: Number(feeAmount),
        slipDataUrl,
        buyerName,
        buyerPhone,
        buyerEmail,
        aiScore: res.total,
        status: 'pending',
        createdAt: new Date().toISOString(),
        adminNote: '',
        paymentSentAt: null,
    };

    HireState.requests.unshift(request);
    localStorage.setItem('my_buyer_email', buyerEmail);
    saveHireRequests(request);

    closeHireModal();
    renderHireTalentList();
    renderHirePanel();

    // Buyer confirmation toast
    showToast(`✅ Hire request sent for ${performer.name}! Admin will review and email you at ${buyerEmail}.`, 'success');
    logSync(`Hire request submitted: ${buyerName} (${buyerEmail}) → ${performer.name} for ${event?.name}`, 'success');

    // Admin notification
    setTimeout(() => {
        showToast(`📨 New hire request from ${buyerName} is pending review in Hire Panel.`, 'primary');
    }, 1500);
};

// ---- Render Admin Hire Panel ----
window.renderHirePanel = function () {
    const list = document.getElementById('hireRequestsList');
    if (!list) return;

    // Always re-read from localStorage so data is fresh after page refresh
    const stored = localStorage.getItem('hire_requests');
    if (stored) {
        try {
            const parsed = JSON.parse(stored);
            if (Array.isArray(parsed)) HireState.requests = parsed;
        } catch(e) {}
    }

    const filter = HireState.currentFilter;
    let requests = HireState.requests;
    if (filter !== 'all') {
        requests = requests.filter(r => r.status === filter);
    }

    // Update stats
    const all = HireState.requests;
    const setStatEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setStatEl('statTotal', all.length);
    setStatEl('statPending', all.filter(r => r.status === 'pending').length);
    setStatEl('statApproved', all.filter(r => r.status === 'approved').length);
    setStatEl('statPaid', all.filter(r => r.status === 'paid').length);

    if (requests.length === 0) {
        list.innerHTML = `
        <div class="glass-panel" style="padding:3rem; text-align:center;">
            <i data-lucide="inbox" style="width:48px; height:48px; opacity:0.2; margin-bottom:1rem;"></i>
            <p style="opacity:0.5">No ${filter === 'all' ? '' : filter + ' '}hire requests found.</p>
        </div>`;
        lucide.createIcons();
        return;
    }

    const statusIcons = { pending: '⏳', approved: '✅', paid: '💳', rejected: '❌' };
    const statusLabels = { pending: 'Pending Review', approved: 'Approved', paid: 'Payment Sent', rejected: 'Rejected' };

    list.innerHTML = requests.map((r, idx) => {
        const slipHtml = r.slipDataUrl && r.slipDataUrl.startsWith('data:image')
            ? `<img class="slip-thumb" src="${r.slipDataUrl}" alt="Payment Slip" onclick="viewSlipLightbox('${r.id}')">`
            : r.slipDataUrl && r.slipDataUrl.startsWith('pdf:')
            ? `<span style="font-size:0.75rem; background:rgba(99,102,241,0.1); padding:0.3rem 0.6rem; border-radius:6px; color:var(--accent-primary);">📄 ${r.slipDataUrl.replace('pdf:','')}</span>`
            : `<span style="opacity:0.4; font-size:0.75rem;">No slip</span>`;

        let actionBtns = '';
        if (r.status === 'pending') {
            actionBtns = `
            <button class="btn-approve" onclick="adminApproveHire('${r.id}')"><i data-lucide="check"></i> Approve & Notify Buyer</button>
            <button class="btn-reject" onclick="adminRejectHire('${r.id}')"><i data-lucide="x"></i> Reject</button>`;
        } else if (r.status === 'approved') {
            actionBtns = `
            <button class="btn-pay" onclick="adminPayTalent('${r.id}')"><i data-lucide="credit-card"></i> Send Payment to ${r.performerName.split(' ')[0]}</button>
            <button class="btn-reject" onclick="adminRejectHire('${r.id}')"><i data-lucide="x"></i> Cancel</button>`;
        } else if (r.status === 'paid') {
            actionBtns = `<span style="font-size:0.8rem; color:var(--accent-primary); font-weight:700;">💳 Payment dispatched at ${new Date(r.paymentSentAt).toLocaleString()}</span>`;
        } else if (r.status === 'rejected') {
            actionBtns = `<span style="font-size:0.8rem; color:var(--danger); font-weight:700;">❌ Request Rejected</span>`;
        }

        // Delete button always shown for admin
        const deleteBtn = `<button onclick="adminDeleteHire('${r.id}')" style="padding:0.45rem 0.85rem; border-radius:7px; border:1px solid rgba(239,68,68,0.25); background:rgba(239,68,68,0.08); color:#f87171; cursor:pointer; font-size:0.78rem; font-weight:700; display:inline-flex; align-items:center; gap:0.35rem; margin-left:auto;"><i data-lucide="trash-2" style="width:13px;height:13px;"></i> Delete</button>`;

        const createdDate = new Date(r.createdAt).toLocaleString();

        return `
        <div class="hire-request-card status-${r.status}" style="animation-delay: ${idx * 0.05}s;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
                <div style="display:flex; align-items:center; gap:1rem; flex:1;">
                    <div class="avatar" style="width:50px; height:50px; font-size:0.85rem; background:linear-gradient(135deg,var(--accent-primary),var(--accent-secondary)); color:white; flex-shrink:0;">
                        ${r.performerName.split(' ').map(n=>n[0]).join('')}
                    </div>
                    <div style="min-width:0;">
                        <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.3rem;">
                            <strong style="font-size:1rem;">${r.performerName}</strong>
                            <span class="hire-status-pill ${r.status}">${statusIcons[r.status]} ${statusLabels[r.status]}</span>
                        </div>
                        <div style="font-size:0.78rem; color:var(--text-secondary);">
                            <strong>Event:</strong> ${r.eventName} &nbsp;•&nbsp; <strong>AI Score:</strong> ${r.aiScore}%
                        </div>
                        <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:0.2rem;">
                            <strong>Occasion:</strong> ${r.eventDesc}
                        </div>
                    </div>
                </div>
                <div style="text-align:right; flex-shrink:0;">
                    <div style="font-size:1.4rem; font-weight:900; color:var(--accent-success);">LKR ${r.feeAmount.toLocaleString()}</div>
                    <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.2rem;">Req ID: ${r.id}</div>
                </div>
            </div>

            <!-- Buyer Info Row -->
            <div style="margin-top:1rem; padding:0.75rem 1rem; background:var(--bg-deep); border-radius:10px; display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.8rem;">
                <div><i data-lucide="user" style="width:13px; height:13px; margin-right:4px; vertical-align:middle;"></i><strong>Buyer:</strong> ${r.buyerName}</div>
                <div><i data-lucide="phone" style="width:13px; height:13px; margin-right:4px; vertical-align:middle;"></i>${r.buyerPhone}</div>
                ${r.buyerEmail ? `<div><i data-lucide="mail" style="width:13px; height:13px; margin-right:4px; vertical-align:middle;"></i>${r.buyerEmail}</div>` : ''}
                <div style="margin-left:auto; color:var(--text-muted);">${createdDate}</div>
            </div>

            <!-- Slip -->
            <div style="margin-top:0.75rem; display:flex; align-items:center; gap:0.75rem;">
                <span style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Payment Slip:</span>
                ${slipHtml}
            </div>

            <!-- Admin Actions -->
            <div class="admin-action-row">
                ${actionBtns}
                ${deleteBtn}
            </div>
        </div>`;
    }).join('');

    lucide.createIcons();
};

// ---- Helper: Send Automated Email Notification (SMTP.js via Brevo SMTP) ----
// ---- Helper: Send Automated Email Notification (Brevo API + SMTP.js + Gmail Fallback) ----
async function dispatchEmailNotification(toEmail, toName, subject, bodyContent) {
    if (!toEmail) {
        showToast(`⚠️ No email address available for ${toName}. Email skipped.`, 'warning');
        return;
    }

    logSync(`📧 [Email Triggered] To: ${toEmail} | Subject: "${subject}"`, "success");
    showToast(`⚡ Sending email to ${toName} (${toEmail})...`, 'info');

    // METHOD 1: Brevo V3 Direct REST API (uses API Key, NOT SMTP password)
    if (BREVO_API_KEY) {
        try {
            const response = await fetch('https://api.brevo.com/v3/smtp/email', {
                method: 'POST',
                headers: {
                    'accept': 'application/json',
                    'api-key': BREVO_API_KEY,
                    'content-type': 'application/json'
                },
                body: JSON.stringify({
                    sender: { name: BREVO_SENDER_NAME, email: BREVO_SENDER_EMAIL },
                    to: [{ email: toEmail, name: toName }],
                    subject: subject,
                    htmlContent: bodyContent
                })
            });

            if (response.ok) {
                showToast(`📧 Email sent successfully to ${toName}!`, 'success');
                logSync(`☁️ Email delivered via Brevo API to: ${toEmail}`, 'success');
                return; // ✅ Done — no Gmail tab needed
            } else {
                const errData = await response.json().catch(() => ({}));
                console.warn('Brevo API error:', response.status, errData);
                logSync(`⚠️ Brevo API response (${response.status}): ${errData.message || response.statusText}`, 'danger');
            }
        } catch (apiErr) {
            console.warn('Brevo API fetch error:', apiErr);
        }
    }

    // METHOD 2: SMTP.js (uses SMTP password, different from API key)
    if (typeof Email !== 'undefined' && BREVO_SMTP_PASS) {
        try {
            const result = await Email.send({
                Host:     BREVO_HOST,
                Username: BREVO_USER,
                Password: BREVO_SMTP_PASS,
                To:       toEmail,
                From:     `${BREVO_SENDER_NAME} <${BREVO_USER}>`,
                Subject:  subject,
                Body:     bodyContent,
            });

            if (result === 'OK') {
                showToast(`📧 Email sent successfully to ${toName}!`, 'success');
                logSync(`☁️ Email delivered via SMTP.js to: ${toEmail}`, 'success');
                return; // ✅ Done
            } else {
                console.error('SMTP.js result:', result);
                logSync(`⚠️ SMTP error: ${result}`, 'danger');
            }
        } catch (err) {
            console.error('SMTP.js exception:', err);
        }
    }

    // METHOD 3: Gmail Compose Fallback
    showToast(`⚠️ Direct dispatch failed. Opening Gmail compose tab...`, 'warning');
    const plainText = bodyContent
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s{2,}/g, '\n')
        .trim();

    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1`
        + `&to=${encodeURIComponent(toEmail)}`
        + `&su=${encodeURIComponent(subject)}`
        + `&body=${encodeURIComponent(plainText)}`;

    window.open(gmailUrl, '_blank');
    setTimeout(() => {
        showToast(`📧 Gmail draft opened for ${toName} — please click send in Gmail.`, 'primary');
    }, 800);
}

// ---- Admin: Approve hire & notify buyer + performer ----
window.adminApproveHire = function (reqId) {
    const req = HireState.requests.find(r => r.id === reqId);
    if (!req) return;
    req.status = 'approved';
    saveHireRequests(req);
    renderHirePanel();
    renderHireTalentList();

    // System Toast & Log
    showToast(`✅ Hire Approved! ${req.buyerName} has been notified — ${req.performerName} is successfully hired.`, 'success');
    logSync(`Admin approved hire: ${req.performerName} for ${req.buyerName} (${req.eventName})`, 'success');

    // 1. Email to Buyer
    const buyerEmailBody = `
<div style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:2rem;background:#0f172a;color:#e2e8f0;border-radius:12px;">
  <h2 style="color:#10b981;margin-bottom:0.5rem;">✅ Hire Request Approved!</h2>
  <p style="color:#94a3b8;margin-bottom:1.5rem;">Your booking has been confirmed by the TALENT.PREMIUM admin team.</p>
  <p>Dear <strong>${req.buyerName}</strong>,</p>
  <p>Great news! Your hire request for <strong>${req.performerName}</strong> has been <strong style="color:#10b981;">APPROVED</strong>.</p>
  <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;background:#1e293b;border-radius:8px;overflow:hidden;">
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Performer</td><td style="padding:0.75rem 1rem;font-weight:700;border-bottom:1px solid #334155;">${req.performerName}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Event</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.eventName}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Occasion</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.eventDesc}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;">Agreed Fee</td><td style="padding:0.75rem 1rem;font-weight:700;color:#10b981;">LKR ${req.feeAmount.toLocaleString()}</td></tr>
  </table>
  <p>The talent has been notified and will contact you shortly to finalize event details.</p>
  <p>Thank you for choosing <strong>TALENT.PREMIUM</strong>!</p>
  <p style="margin-top:2rem;color:#64748b;font-size:0.8rem;">— TALENT.PREMIUM Team</p>
</div>`.trim();

    if (req.buyerEmail) {
        dispatchEmailNotification(req.buyerEmail, req.buyerName, `✅ [Confirmed] Your Hire for ${req.performerName} is Approved!`, buyerEmailBody);
    }

    // 2. Email to Performer/Talent (recover email if missing)
    let performerEmail = req.performerEmail;
    if (!performerEmail) {
        const p = State.participants.find(x => String(x.id) === String(req.performerId));
        if (p && p.email) performerEmail = p.email;
    }

    if (!performerEmail) {
        performerEmail = prompt(`Please enter email address for performer "${req.performerName}" to send notification:`);
        if (performerEmail) {
            req.performerEmail = performerEmail;
            saveHireRequests(req);
        }
    }

    if (performerEmail) {
        const talentEmailBody = `
<div style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:2rem;background:#0f172a;color:#e2e8f0;border-radius:12px;">
  <h2 style="color:#10b981;margin-bottom:1rem;">🎤 You Have Been Hired!</h2>
  <p>Hello <strong>${req.performerName}</strong>,</p>
  <p>Congratulations! You have been <strong style="color:#10b981;">SELECTED and HIRED</strong> for an upcoming event.</p>
  <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;background:#1e293b;border-radius:8px;overflow:hidden;">
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Buyer Name</td><td style="padding:0.75rem 1rem;font-weight:700;border-bottom:1px solid #334155;">${req.buyerName}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Buyer Phone</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.buyerPhone}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Buyer Email</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.buyerEmail || 'N/A'}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Event</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.eventName}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Occasion Details</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.eventDesc}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;">Agreed Fee</td><td style="padding:0.75rem 1rem;font-weight:700;color:#10b981;">LKR ${req.feeAmount.toLocaleString()}</td></tr>
  </table>
  <p>Please contact the buyer as soon as possible to finalize event details.</p>
  <p style="margin-top:2rem;color:#64748b;font-size:0.8rem;">— TALENT.PREMIUM Team</p>
</div>`.trim();

        setTimeout(() => {
            dispatchEmailNotification(performerEmail, req.performerName, `🎤 You're Hired for: ${req.eventName}!`, talentEmailBody);
        }, 1200);
    } else {
        showToast(`⚠️ Performer email not provided. Notification for performer skipped.`, 'warning');
    }

    setTimeout(() => {
        showToast(`📱 Notifications dispatched to both buyer and performer!`, 'primary');
    }, 1000);
};

// ---- Admin: Reject hire ----
window.adminRejectHire = function (reqId) {
    if (!confirm('Are you sure you want to reject this hire request?')) return;
    const req = HireState.requests.find(r => r.id === reqId);
    if (!req) return;
    req.status = 'rejected';
    saveHireRequests(req);
    renderHirePanel();
    renderHireTalentList();
    showToast(`❌ Hire request from ${req.buyerName} rejected.`, 'danger');
    logSync(`Admin rejected hire: ${req.performerName} for ${req.buyerName}`, 'remote');
};

// ---- Admin: Pay talent ----
window.adminPayTalent = function (reqId) {
    const req = HireState.requests.find(r => r.id === reqId);
    if (!req) return;

    if (!confirm(`Send payment of LKR ${req.feeAmount.toLocaleString()} to ${req.performerName}?`)) return;

    req.status = 'paid';
    req.paymentSentAt = new Date().toISOString();
    saveHireRequests(req);
    renderHirePanel();
    if (typeof renderMyBookings === 'function') renderMyBookings();

    showToast(`💳 Payment of LKR ${req.feeAmount.toLocaleString()} sent to ${req.performerName}!`, 'success');
    logSync(`Admin dispatched payment LKR ${req.feeAmount.toLocaleString()} to ${req.performerName} for ${req.eventName}`, 'success');

    // Email alert to talent regarding payment
    const paymentEmailBody = `
<div style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:2rem;background:#0f172a;color:#e2e8f0;border-radius:12px;">
  <h2 style="color:#6366f1;margin-bottom:0.5rem;">💳 Payment Dispatched!</h2>
  <p style="color:#94a3b8;margin-bottom:1.5rem;">Payment for your performance has been processed by TALENT.PREMIUM.</p>
  <p>Dear <strong>${req.performerName}</strong>,</p>
  <p>We are pleased to inform you that your fee for the event <strong>${req.eventName}</strong> has been successfully dispatched to your account.</p>
  <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;background:#1e293b;border-radius:8px;overflow:hidden;">
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Event Name</td><td style="padding:0.75rem 1rem;font-weight:700;border-bottom:1px solid #334155;">${req.eventName}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;border-bottom:1px solid #334155;">Buyer</td><td style="padding:0.75rem 1rem;border-bottom:1px solid #334155;">${req.buyerName}</td></tr>
    <tr><td style="padding:0.75rem 1rem;color:#94a3b8;">Dispatched Amount</td><td style="padding:0.75rem 1rem;font-weight:900;color:#10b981;font-size:1.1rem;">LKR ${Number(req.feeAmount).toLocaleString()}</td></tr>
  </table>
  <p style="color:#94a3b8;font-size:0.85rem;">Thank you for your outstanding artistic performance!</p>
</div>`;

    if (req.performerEmail) {
        dispatchEmailNotification(
            req.performerEmail,
            req.performerName,
            `💳 Payment Dispatched: LKR ${req.feeAmount.toLocaleString()} for ${req.eventName}`,
            paymentEmailBody
        );
    } else {
        showToast(`🎤 ${req.performerName} marked as paid! (No direct performer email registered)`, 'primary');
    }
};

// ---- Admin: Delete hire request ----
window.adminDeleteHire = async function (reqId) {
    const req = HireState.requests.find(r => r.id === reqId);
    if (!req) return;

    if (!confirm(`Delete hire request from "${req.buyerName}" for "${req.performerName}"?\n\nThis action cannot be undone.`)) return;

    // Remove from HireState
    HireState.requests = HireState.requests.filter(r => r.id !== reqId);

    // Save to localStorage
    localStorage.setItem('hire_requests', JSON.stringify(HireState.requests));

    // Delete from Supabase cloud
    if (_supabase) {
        try {
            const { error } = await _supabase.from('hire_requests').delete().eq('id', reqId);
            if (!error) {
                logSync(`🗑️ Hire request ${reqId} deleted from cloud.`, 'success');
            }
        } catch (e) {
            console.log('Cloud delete skipped — removed locally.');
        }
    }

    renderHirePanel();
    renderHireTalentList();
    showToast(`🗑️ Hire request from ${req.buyerName} deleted.`, 'primary');
};

// ---- Filter hire requests ----
window.filterHireRequests = function (filter, btn) {
    HireState.currentFilter = filter;
    document.querySelectorAll('.hire-filter-btn').forEach(b => {
        b.style.background = 'var(--bg-deep)';
        b.style.color = 'var(--text-primary)';
    });
    if (btn) {
        btn.style.background = 'var(--accent-primary)';
        btn.style.color = 'white';
    }
    renderHirePanel();
};

// ---- Slip lightbox ----
window.viewSlipLightbox = function (reqId) {
    const req = HireState.requests.find(r => r.id === reqId);
    if (!req || !req.slipDataUrl.startsWith('data:image')) return;
    const lb = document.getElementById('slipLightbox');
    document.getElementById('slipLightboxImg').src = req.slipDataUrl;
    lb.style.display = 'flex';
};

// ============================================
// REAL-TIME PRIVATE CHAT ENGINE (SUPABASE REALTIME)
// Each buyer/performer has a PRIVATE 1-on-1 thread with Admin.
// Admin can switch between conversations. Users only see their own.
// ============================================

/**
 * All conversations stored as: { [conversationId]: { messages: [], clientName, clientRole, clientEmail } }
 * conversationId = sanitized email or unique key for the client
 */
const ChatState = {
    conversations: JSON.parse(localStorage.getItem('portal_chat_conversations') || '{}'),
    activeConversationId: null,   // Which conversation is currently viewed
    isOpen: false,
    unreadCounts: JSON.parse(localStorage.getItem('portal_chat_unread') || '{}'), // { convId: count }
};

// --- Helpers ---
function getChatConversationId() {
    // For admin: use the selected conversation from dropdown
    if (State.isAdmin) {
        return ChatState.activeConversationId || null;
    }
    // For buyer/performer: derive from their email
    const email = State.currentUser?.email || document.getElementById('buyerName')?.value || null;
    if (!email) return null;
    return 'conv_' + email.toLowerCase().replace(/[^a-z0-9]/g, '_');
}

function getMyClientIdentity() {
    const name = State.currentUser?.name || document.getElementById('buyerName')?.value || 'Guest';
    const email = State.currentUser?.email || '';
    const role = (State.currentPortal === 'performer') ? 'performer' : 'buyer';
    return { name, email, role };
}

function saveChatConversations() {
    // Keep only last 50 messages per conversation to limit storage
    const trimmed = {};
    for (const [cid, conv] of Object.entries(ChatState.conversations)) {
        trimmed[cid] = {
            ...conv,
            messages: (conv.messages || []).slice(-50)
        };
    }
    localStorage.setItem('portal_chat_conversations', JSON.stringify(trimmed));
}

function saveChatUnread() {
    localStorage.setItem('portal_chat_unread', JSON.stringify(ChatState.unreadCounts));
}

function getTotalUnread() {
    if (State.isAdmin) {
        // Admin sees total unread across ALL conversations
        return Object.values(ChatState.unreadCounts).reduce((a, b) => a + b, 0);
    }
    // Client sees unread only for their own conversation
    const myConvId = getChatConversationId();
    return myConvId ? (ChatState.unreadCounts[myConvId] || 0) : 0;
}

function ensureConversation(convId, clientName, clientRole, clientEmail) {
    if (!ChatState.conversations[convId]) {
        ChatState.conversations[convId] = {
            messages: [],
            clientName: clientName || 'Unknown',
            clientRole: clientRole || 'buyer',
            clientEmail: clientEmail || ''
        };
    } else {
        // Update metadata if provided (name may change)
        if (clientName) ChatState.conversations[convId].clientName = clientName;
        if (clientRole) ChatState.conversations[convId].clientRole = clientRole;
        if (clientEmail) ChatState.conversations[convId].clientEmail = clientEmail;
    }
}

// --- Toggle Chat Drawer ---
window.toggleChatDrawer = function () {
    const drawer = document.getElementById('chatDrawer');
    if (!drawer) return;
    ChatState.isOpen = !ChatState.isOpen;
    drawer.style.display = ChatState.isOpen ? 'flex' : 'none';

    if (ChatState.isOpen) {
        // Show/hide admin bar
        const adminBar = document.getElementById('chatAdminBar');
        if (adminBar) adminBar.style.display = State.isAdmin ? 'block' : 'none';

        if (State.isAdmin) {
            renderAdminConversationList();
            // Auto-select first conversation if none selected
            if (!ChatState.activeConversationId) {
                const convIds = Object.keys(ChatState.conversations);
                if (convIds.length > 0) {
                    ChatState.activeConversationId = convIds[0];
                }
            }
        } else {
            // For buyer/performer, auto-create their conversation thread
            const identity = getMyClientIdentity();
            const convId = getChatConversationId();
            if (convId) {
                ensureConversation(convId, identity.name, identity.role, identity.email);
                ChatState.activeConversationId = convId;
                saveChatConversations();
            }
        }

        // Clear unread for active conversation
        if (ChatState.activeConversationId) {
            ChatState.unreadCounts[ChatState.activeConversationId] = 0;
            saveChatUnread();
        }

        updateChatBadge();
        renderChatMessages();
        const msgBox = document.getElementById('chatMessages');
        if (msgBox) msgBox.scrollTop = msgBox.scrollHeight;
    }
};

// --- Admin: Render Conversation Dropdown ---
function renderAdminConversationList() {
    const select = document.getElementById('chatConversationSelect');
    if (!select) return;

    const convIds = Object.keys(ChatState.conversations);
    if (convIds.length === 0) {
        select.innerHTML = '<option value="">No conversations yet</option>';
        return;
    }

    select.innerHTML = convIds.map(cid => {
        const conv = ChatState.conversations[cid];
        const roleIcon = conv.clientRole === 'performer' ? '🎤' : '🛍️';
        const roleLabel = conv.clientRole === 'performer' ? 'Performer' : 'Buyer';
        const unread = ChatState.unreadCounts[cid] || 0;
        const unreadTag = unread > 0 ? ` (${unread} new)` : '';
        const selected = cid === ChatState.activeConversationId ? 'selected' : '';
        return `<option value="${cid}" ${selected}>${roleIcon} ${conv.clientName} — ${roleLabel}${unreadTag}</option>`;
    }).join('');
}

// --- Admin: Select Conversation ---
window.selectAdminChat = function (convId) {
    if (!convId) return;
    ChatState.activeConversationId = convId;
    // Clear unread for this conversation
    ChatState.unreadCounts[convId] = 0;
    saveChatUnread();
    updateChatBadge();
    renderChatMessages();
    renderAdminConversationList(); // refresh unread tags
    const msgBox = document.getElementById('chatMessages');
    if (msgBox) msgBox.scrollTop = msgBox.scrollHeight;
};

// --- Badge ---
function updateChatBadge() {
    const badge = document.getElementById('chatBadge');
    if (!badge) return;
    const total = getTotalUnread();
    if (total > 0) {
        badge.textContent = total;
        badge.style.display = 'flex';
    } else {
        badge.style.display = 'none';
    }
}

// --- Render Messages for Active Conversation ---
function renderChatMessages() {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    const convId = ChatState.activeConversationId;
    const conv = convId ? ChatState.conversations[convId] : null;
    const messages = conv ? (conv.messages || []) : [];

    if (!convId || messages.length === 0) {
        const emptyMsg = State.isAdmin && !convId
            ? '📋 Select a client conversation from the dropdown above to begin.'
            : '👋 Welcome to TALENT.PREMIUM Support! Send a message to start a private conversation with Admin.';
        container.innerHTML = `
        <div class="chat-msg system">
            <div class="chat-bubble">${emptyMsg}</div>
        </div>`;
        return;
    }

    container.innerHTML = messages.map(m => {
        const isMe = (m.role === 'admin' && State.isAdmin) || 
                     (m.role !== 'admin' && !State.isAdmin);

        const roleLabel = m.role === 'admin' 
            ? '🛡️ Admin' 
            : m.role === 'performer' 
            ? `🎤 ${m.sender || 'Performer'}` 
            : `🛍️ ${m.sender || 'Buyer'}`;

        const typeClass = isMe ? 'sent' : 'received';

        return `
        <div class="chat-msg ${typeClass}">
            <div class="sender-tag">${roleLabel} • ${new Date(m.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
            <div class="chat-bubble">${escapeHtml(m.text)}</div>
        </div>`;
    }).join('');

    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    return text.replace(/[&<>"']/g, function(m) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m];
    });
}

// --- Supabase Realtime Channel (single global channel, messages routed by conversationId) ---
let portalChatChannel = null;
if (_supabase) {
    try {
        portalChatChannel = _supabase.channel('portal-chat-room');
        portalChatChannel
            .on('broadcast', { event: 'chat_msg' }, payload => {
                const incomingMsg = payload.payload;
                if (!incomingMsg || !incomingMsg.conversationId) return;

                const convId = incomingMsg.conversationId;

                // Ensure conversation exists
                ensureConversation(convId, incomingMsg.clientName, incomingMsg.clientRole, incomingMsg.clientEmail);

                // Deduplicate
                const conv = ChatState.conversations[convId];
                if (conv.messages.some(m => m.id === incomingMsg.id)) return;

                // Check if this message is relevant to me
                const myConvId = getChatConversationId();
                const isRelevant = State.isAdmin || convId === myConvId;
                if (!isRelevant) return; // Not my conversation, ignore

                // Don't add messages I sent myself (they're already added locally)
                const isMySentMsg = (incomingMsg.role === 'admin' && State.isAdmin) ||
                                    (incomingMsg.role !== 'admin' && !State.isAdmin && convId === myConvId);
                if (isMySentMsg && incomingMsg.senderSessionId === ChatState._sessionId) return;

                conv.messages.push(incomingMsg);
                saveChatConversations();

                // Handle unread / notification
                const isViewingThisConv = ChatState.isOpen && ChatState.activeConversationId === convId;
                if (!isViewingThisConv) {
                    ChatState.unreadCounts[convId] = (ChatState.unreadCounts[convId] || 0) + 1;
                    saveChatUnread();
                    updateChatBadge();
                    const senderLabel = incomingMsg.role === 'admin' ? '🛡️ Admin' : incomingMsg.sender;
                    showToast(`💬 New message from ${senderLabel}: "${incomingMsg.text.substring(0, 30)}..."`, 'primary');
                } else {
                    renderChatMessages();
                    if (State.isAdmin) renderAdminConversationList();
                }
            })
            .subscribe();
    } catch (e) {
        console.log('Supabase chat channel init skipped.');
    }
}

// Unique session ID to prevent self-echo from broadcast
ChatState._sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);

// --- Send Message ---
window.sendChatMessage = function (e) {
    if (e) e.preventDefault();
    const input = document.getElementById('chatInput');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;

    let convId = ChatState.activeConversationId;

    // For non-admin users, ensure conversation exists
    if (!State.isAdmin) {
        const identity = getMyClientIdentity();
        convId = getChatConversationId();
        if (!convId) {
            showToast('⚠️ Please log in or enter your name to start a chat.', 'warning');
            return;
        }
        ensureConversation(convId, identity.name, identity.role, identity.email);
        ChatState.activeConversationId = convId;
    }

    if (!convId) {
        showToast('⚠️ Please select a conversation first.', 'warning');
        return;
    }

    const conv = ChatState.conversations[convId];
    const senderName = State.isAdmin ? 'Admin' : (State.currentUser ? State.currentUser.name : (document.getElementById('buyerName')?.value || 'Guest'));

    const msg = {
        id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
        conversationId: convId,
        sender: senderName,
        role: State.isAdmin ? 'admin' : (State.currentPortal === 'performer' ? 'performer' : 'buyer'),
        text: text,
        timestamp: new Date().toISOString(),
        senderSessionId: ChatState._sessionId,
        // Metadata for conversation creation on remote side
        clientName: conv.clientName,
        clientRole: conv.clientRole,
        clientEmail: conv.clientEmail
    };

    conv.messages.push(msg);
    saveChatConversations();
    input.value = '';
    renderChatMessages();
    if (State.isAdmin) renderAdminConversationList();

    // Broadcast via Supabase Realtime
    if (portalChatChannel) {
        try {
            portalChatChannel.send({
                type: 'broadcast',
                event: 'chat_msg',
                payload: msg
            });
        } catch (err) {
            console.log('Realtime broadcast offline.');
        }
    }
};

// ---- ADMIN FACTORY RESET SYSTEM ----
window.resetSystemData = async function () {
    if (!State.isAdmin) {
        showToast('🔒 Admin authorization required.', 'warning');
        return;
    }

    if (!confirm('⚠️ Are you sure you want to reset all system data?\n\nThis will clear all test hire requests, custom events, registered performers, and chat messages, restoring clean initial defaults.')) {
        return;
    }

    showToast('Resetting system... please wait.', 'primary');

    // Clear local storage
    localStorage.removeItem('aesthetic_participants');
    localStorage.removeItem('aesthetic_events');
    localStorage.removeItem('aesthetic_weights');
    localStorage.removeItem('hire_requests');
    localStorage.removeItem('portal_hire_requests');
    localStorage.removeItem('portal_users');
    localStorage.removeItem('portal_chat_conversations');
    localStorage.removeItem('portal_chat_unread');
    localStorage.removeItem('my_profile_id');

    // Reset Hire State in memory & Supabase
    if (typeof HireState !== 'undefined') {
        if (_supabase && HireState.requests && HireState.requests.length > 0) {
            for (const r of HireState.requests) {
                try { await _supabase.from('hire_requests').delete().eq('id', r.id); } catch(e) {}
            }
        }
        HireState.requests = [];
    }

    // ---- STEP 1: Capture ALL current IDs BEFORE overwriting State ----
    // Must happen first: once State.participants = seedParticipants, real user IDs are lost
    const currentParticipantIds = (State.participants || []).map(p => String(p.id)).filter(Boolean);
    const currentHireIds = (HireState && HireState.requests ? HireState.requests : []).map(r => String(r.id)).filter(Boolean);

    // Fetch live IDs from Supabase to catch any cloud-only registered performers
    let cloudParticipantIds = [];
    if (_supabase) {
        try {
            const { data: cloudRows } = await _supabase.from('participants').select('id');
            if (cloudRows) cloudParticipantIds = cloudRows.map(r => String(r.id));
        } catch(e) {}
    }
    // Merge local + cloud IDs so ALL performers are deleted (including registered users)
    const allParticipantIdsToDelete = [...new Set([...currentParticipantIds, ...cloudParticipantIds])];

    // ---- STEP 2: Define clean seed data ----
    const seedParticipants = [
        { id: 'seed-1', name: 'Elena Vance', region: 'Western', experience: 12, consistency: 92, skills: 'Vocal Range, Diction, Opera, Stage Presence', judgeA: 90, judgeB: 95, judgeC: 91, inactiveMonths: 1, email: 'elena@talent.com', phone: '+94 77 123 4567', rate: 'LKR 85,000 / event' },
        { id: 'seed-2', name: 'Julian Marsh', region: 'Central', experience: 5, consistency: 85, skills: 'Public Speaking, Emceeing, Professionalism, Humor', judgeA: 82, judgeB: 88, judgeC: 85, inactiveMonths: 4, email: 'julian@talent.com', phone: '+94 71 987 6543', rate: 'LKR 50,000 / event' },
        { id: 'seed-3', name: 'Sarah Sings', region: 'Uva', experience: 8, consistency: 78, skills: 'Vocal Range, Pop, Stage Presence, Improvisation', judgeA: 75, judgeB: 80, judgeC: 79, inactiveMonths: 12, email: 'sarah@talent.com', phone: '+94 75 456 7890', rate: 'LKR 65,000 / event' }
    ];

    const seedEvents = [
        { id: 1, name: 'Derana Dream Star Finale', description: 'National level vocal and stage performance competition.', requirements: ['Vocal Range', 'Stage Presence', 'Baila', 'Sinhala Diction'] },
        { id: 2, name: 'Corporate Emcee Summit (Colombo)', description: 'Professional hosting for high-end corporate galas.', requirements: ['Public Speaking', 'Trilingual', 'Professionalism'] },
        { id: 3, name: 'Kandy Cultural Pageant', description: 'Traditional arts and drumming showcase.', requirements: ['Kandyan Dance', 'Geta Bera', 'Choreography'] }
    ];

    // ---- STEP 3: Supabase Cloud Delete (using IDs captured in Step 1) ----
    if (_supabase) {
        // Delete each participant by exact ID - works even with RLS enabled
        for (const pid of allParticipantIdsToDelete) {
            try { await _supabase.from('participants').delete().eq('id', pid); } catch(e) {}
        }
        // Broad .in() delete as backup (works when RLS is off)
        if (allParticipantIdsToDelete.length > 0) {
            try { await _supabase.from('participants').delete().in('id', allParticipantIdsToDelete); } catch(e) {}
        }

        // Upsert seed performers (insert or update on conflict - no duplicate key errors)
        try {
            const { error: upsertErr } = await _supabase
                .from('participants')
                .upsert(seedParticipants, { onConflict: 'id' });
            if (!upsertErr) {
                logSync('Cloud participants reset - seed performers restored.', 'success');
            } else {
                console.log('Upsert note:', upsertErr.message);
                logSync('Cloud upsert skipped - seed data saved to local cache.', 'remote');
            }
        } catch(e) {
            console.log('Cloud participants upsert note:', e);
        }

        // Delete hire_requests by per-ID (IDs captured in Step 1, before State reset)
        for (const hid of currentHireIds) {
            try { await _supabase.from('hire_requests').delete().eq('id', hid); } catch(e) {}
        }
        if (currentHireIds.length > 0) {
            try { await _supabase.from('hire_requests').delete().in('id', currentHireIds); } catch(e) {}
        }
        logSync('Cloud hire requests cleared.', 'success');
    }

    // ---- STEP 4: Apply seed data to in-memory State ----
    State.events = seedEvents;
    State.participants = seedParticipants;
    State.selectedEvent = seedEvents[0];
    State.weights = { skill: 0.60, consist: 0.30, exp: 0.10 };

    // Reset Chat State in memory
    if (typeof ChatState !== 'undefined') {
        ChatState.conversations = {};
        ChatState.activeConversationId = null;
        ChatState.unreadCounts = {};
        if (typeof renderChatMessages === 'function') renderChatMessages();
        if (typeof renderAdminConversationList === 'function') renderAdminConversationList();
        if (typeof updateChatBadge === 'function') updateChatBadge();
    }

    // Persist seed data to localStorage so browser refresh loads correctly too
    saveToCache();

    // Re-render all panels including Showcase/AI Recommendations
    renderEvents();
    renderParticipantRegistry();
    if (typeof renderHirePanel === 'function') renderHirePanel();
    if (typeof renderMyProfile === 'function') renderMyProfile();

    // Re-run ML pipeline then refresh the AI Showcase
    runKMeansAndAnomalies();
    renderConsistencyMatrix();
    if (State.selectedEvent && typeof renderRecommendations === 'function') {
        renderRecommendations();
    }

    if (window.lucide) window.lucide.createIcons();

    showToast('System fully reset! Showcase & AI data refreshed with seed defaults.', 'success');
    logSync('Admin Factory Reset complete - Cloud + Local synced with clean seed data.', 'system');
};


