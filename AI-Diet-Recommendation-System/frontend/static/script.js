document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Elements
    const form = e.target;
    const submitBtn = document.getElementById('submit-btn');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    
    // Hide results and show loading
    resultsContainer.classList.add('hidden');
    loading.classList.remove('hidden');
    submitBtn.disabled = true;
    
    // Gather form data
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // --- BMI Calculation ---
    const height = parseFloat(data.Height);
    const weight = parseFloat(data.Weight);
    let bmi = 0;
    let bmiCategory = 'Unknown';
    let bmiClass = '';

    if (height > 0 && weight > 0) {
        bmi = weight / (height * height);
        
        if (bmi < 18.5) {
            bmiCategory = 'Underweight';
            bmiClass = 'bmi-underweight';
        } else if (bmi < 25) {
            bmiCategory = 'Normal';
            bmiClass = 'bmi-normal';
        } else if (bmi < 30) {
            bmiCategory = 'Overweight';
            bmiClass = 'bmi-overweight';
        } else {
            bmiCategory = 'Obese';
            bmiClass = 'bmi-obese';
        }
    }
    
    // Attach BMI data for the backend (used by Gemini, not by the ML model)
    data.bmi = bmi.toFixed(1);
    data.bmi_category = bmiCategory;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok || result.error) {
            throw new Error(result.error || 'Failed to generate prediction');
        }
        
        // --- Populate BMI Card ---
        const bmiCard = document.getElementById('bmi-card');
        const bmiValue = document.getElementById('bmi-value');
        const bmiCategoryEl = document.getElementById('bmi-category');

        // Clear previous BMI color classes
        bmiCard.classList.remove('bmi-underweight', 'bmi-normal', 'bmi-overweight', 'bmi-obese');
        bmiCard.classList.add(bmiClass);
        bmiValue.textContent = bmi.toFixed(1);
        bmiCategoryEl.textContent = bmiCategory;

        // --- Populate Prediction & Goal ---
        document.getElementById('obesity-badge').textContent = formatString(result.obesity_prediction);
        document.getElementById('goal-badge').textContent = result.diet_goal;
        document.getElementById('ai-text').innerHTML = result.ai_explanation || 'No explanation provided.';
        
        // --- Render Meals ---
        renderMeals('breakfast-list', result.meal_plan.breakfast);
        renderMeals('lunch-list', result.meal_plan.lunch);
        renderMeals('dinner-list', result.meal_plan.dinner);
        
        // Show Results
        loading.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
        // Scroll to results smoothly
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        alert('Error: ' + error.message);
        loading.classList.add('hidden');
    } finally {
        submitBtn.disabled = false;
    }
});

function formatString(str) {
    if (!str) return '';
    return str.replace(/_/g, ' ');
}

function renderMeals(elementId, mealsArray) {
    const list = document.getElementById(elementId);
    list.innerHTML = '';
    
    if (!mealsArray || mealsArray.length === 0) {
        list.innerHTML = '<li><span class="food-name" style="color: var(--text-muted)">No specific meals found.</span></li>';
        return;
    }
    
    mealsArray.forEach(food => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="food-name">${food.food_name}</span>
            <div class="macros">
                <span class="macro-tag">🔥 ${Math.round(food.calories)} kcal</span>
                <span class="macro-tag">💪 ${Math.round(food.protein)}g pro</span>
                <span class="macro-tag">🥑 ${Math.round(food.fat)}g fat</span>
            </div>
        `;
        list.appendChild(li);
    });
}
