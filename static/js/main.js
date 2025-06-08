/**
 * Heart Disease Prediction - Main JavaScript File
 * Contains global functionality and utilities
 */

// Global variables
window.HeartApp = window.HeartApp || {};

// Initialize app when DOM is ready
$(document).ready(function() {
    HeartApp.init();
});

// Main application object
HeartApp = {
    // Configuration
    config: {
        animationDuration: 300,
        toastDuration: 5000,
        autoSaveInterval: 30000,
        debounceDelay: 300
    },

    // Initialize the application
    init: function() {
        this.initTooltips();
        this.initPopovers();
        this.initFormValidation();
        this.initProgressTracking();
        this.initNotifications();
        this.initAccessibility();
        this.initErrorHandling();
        this.initAutoSave();
        
        console.log('Heart Disease Prediction App initialized');
    },

    // Initialize Bootstrap tooltips
    initTooltips: function() {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                delay: { show: 500, hide: 100 }
            });
        });
    },

    // Initialize Bootstrap popovers
    initPopovers: function() {
        var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
        var popoverList = popoverTriggerList.map(function(popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl);
        });
    },

    // Form validation enhancements
    initFormValidation: function() {
        // Real-time validation for forms
        $('form').on('input change', 'input, select, textarea', function() {
            HeartApp.validateField($(this));
        });

        // Form submission handling
        $('form').on('submit', function(e) {
            var form = $(this);
            if (!HeartApp.validateForm(form)) {
                e.preventDefault();
                HeartApp.showNotification('Please correct the errors in the form', 'danger');
                return false;
            }
            
            HeartApp.showLoadingState(form);
        });

        // Password strength indicator
        $('input[type="password"]').on('input', function() {
            HeartApp.updatePasswordStrength($(this));
        });
    },

    // Validate individual field
    validateField: function(field) {
        var isValid = true;
        var value = field.val().trim();
        var fieldType = field.attr('type') || field.prop('tagName').toLowerCase();

        // Remove previous validation classes
        field.removeClass('is-valid is-invalid');

        // Required field validation
        if (field.attr('required') && !value) {
            field.addClass('is-invalid');
            return false;
        }

        // Email validation
        if (fieldType === 'email' && value) {
            var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                field.addClass('is-invalid');
                return false;
            }
        }

        // Number validation
        if (fieldType === 'number' && value) {
            var min = field.attr('min');
            var max = field.attr('max');
            var numValue = parseFloat(value);
            
            if (min && numValue < parseFloat(min)) {
                field.addClass('is-invalid');
                return false;
            }
            if (max && numValue > parseFloat(max)) {
                field.addClass('is-invalid');
                return false;
            }
        }

        // Password confirmation
        if (field.attr('name') === 'password2') {
            var password = $('input[name="password"]').val();
            if (value !== password) {
                field.addClass('is-invalid');
                return false;
            }
        }

        // If we get here, field is valid
        if (value) {
            field.addClass('is-valid');
        }
        return true;
    },

    // Validate entire form
    validateForm: function(form) {
        var isValid = true;
        
        form.find('input, select, textarea').each(function() {
            if (!HeartApp.validateField($(this))) {
                isValid = false;
            }
        });

        return isValid;
    },

    // Password strength indicator
    updatePasswordStrength: function(passwordField) {
        var password = passwordField.val();
        var strength = this.calculatePasswordStrength(password);
        var strengthBar = passwordField.siblings('.password-strength');
        
        if (strengthBar.length === 0) {
            strengthBar = $('<div class="password-strength mt-2"></div>');
            passwordField.after(strengthBar);
        }

        var strengthText = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'];
        var strengthClass = ['danger', 'danger', 'warning', 'info', 'success'];
        
        strengthBar.html(`
            <div class="progress" style="height: 4px;">
                <div class="progress-bar bg-${strengthClass[strength]}" 
                     style="width: ${(strength + 1) * 20}%"></div>
            </div>
            <small class="text-${strengthClass[strength]}">
                Password strength: ${strengthText[strength]}
            </small>
        `);
    },

    // Calculate password strength
    calculatePasswordStrength: function(password) {
        var score = 0;
        
        if (password.length >= 8) score++;
        if (/[a-z]/.test(password)) score++;
        if (/[A-Z]/.test(password)) score++;
        if (/[0-9]/.test(password)) score++;
        if (/[^A-Za-z0-9]/.test(password)) score++;
        
        return Math.min(score, 4);
    },

    // Progress tracking for forms
    initProgressTracking: function() {
        $('form').each(function() {
            var form = $(this);
            var progressBar = form.find('.progress-bar');
            
            if (progressBar.length > 0) {
                form.on('input change', function() {
                    HeartApp.updateFormProgress(form, progressBar);
                });
            }
        });
    },

    // Update form progress
    updateFormProgress: function(form, progressBar) {
        var totalFields = form.find('input:not([type="hidden"]), select, textarea').length;
        var filledFields = 0;
        
        form.find('input:not([type="hidden"]), select, textarea').each(function() {
            if ($(this).val().trim() !== '') {
                filledFields++;
            }
        });
        
        var progress = totalFields > 0 ? (filledFields / totalFields) * 100 : 0;
        progressBar.css('width', progress + '%');
        progressBar.attr('aria-valuenow', progress);
    },

    // Notification system
    initNotifications: function() {
        // Create toast container if it doesn't exist
        if ($('#toastContainer').length === 0) {
            $('body').append(`
                <div id="toastContainer" class="toast-container position-fixed top-0 end-0 p-3" 
                     style="z-index: 1100;"></div>
            `);
        }
    },

    // Show notification toast
    showNotification: function(message, type = 'info', title = null) {
        var toastId = 'toast-' + Date.now();
        var iconClass = {
            'success': 'fa-check-circle',
            'danger': 'fa-exclamation-triangle',
            'warning': 'fa-exclamation-circle',
            'info': 'fa-info-circle'
        };

        var toast = $(`
            <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" 
                 role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas ${iconClass[type] || iconClass.info} me-2"></i>
                        ${title ? '<strong>' + title + '</strong><br>' : ''}
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                            data-bs-dismiss="toast"></button>
                </div>
            </div>
        `);

        $('#toastContainer').append(toast);
        
        var bsToast = new bootstrap.Toast(toast[0], {
            autohide: true,
            delay: this.config.toastDuration
        });
        
        bsToast.show();

        // Remove toast element after it's hidden
        toast.on('hidden.bs.toast', function() {
            $(this).remove();
        });
    },

    // Loading state management
    showLoadingState: function(element) {
        var btn = element.find('button[type="submit"]');
        if (btn.length > 0) {
            btn.prop('disabled', true);
            var originalText = btn.html();
            btn.data('original-text', originalText);
            btn.html('<i class="fas fa-spinner fa-spin me-2"></i>Loading...');
        }
        
        element.addClass('loading');
    },

    // Hide loading state
    hideLoadingState: function(element) {
        var btn = element.find('button[type="submit"]');
        if (btn.length > 0) {
            btn.prop('disabled', false);
            var originalText = btn.data('original-text');
            if (originalText) {
                btn.html(originalText);
            }
        }
        
        element.removeClass('loading');
    },

    // Accessibility enhancements
    initAccessibility: function() {
        // Add ARIA labels to form controls without labels
        $('input, select, textarea').each(function() {
            var element = $(this);
            if (!element.attr('aria-label') && !element.attr('aria-labelledby')) {
                var placeholder = element.attr('placeholder');
                var name = element.attr('name');
                if (placeholder) {
                    element.attr('aria-label', placeholder);
                } else if (name) {
                    element.attr('aria-label', name.replace(/[_-]/g, ' '));
                }
            }
        });

        // Keyboard navigation enhancements
        $(document).on('keydown', function(e) {
            // Skip links for screen readers
            if (e.key === 'Tab' && e.shiftKey === false && e.target === document.body) {
                var skipLink = $('#skipToMain');
                if (skipLink.length === 0) {
                    skipLink = $('<a id="skipToMain" href="#main" class="visually-hidden-focusable">Skip to main content</a>');
                    $('body').prepend(skipLink);
                }
            }
        });

        // Focus management for modals
        $(document).on('shown.bs.modal', '.modal', function() {
            $(this).find('input, select, textarea, button').first().focus();
        });
    },

    // Error handling
    initErrorHandling: function() {
        // Global AJAX error handler
        $(document).ajaxError(function(event, xhr, settings, error) {
            console.error('AJAX Error:', error);
            HeartApp.showNotification('An error occurred. Please try again.', 'danger');
        });

        // Global error handler
        window.addEventListener('error', function(e) {
            console.error('JavaScript Error:', e.error);
            // Don't show error notifications for minor issues
            if (e.error && e.error.name !== 'ResizeObserver' && e.error.name !== 'Network') {
                HeartApp.showNotification('An unexpected error occurred.', 'danger');
            }
        });
    },

    // Auto-save functionality
    initAutoSave: function() {
        var autoSaveFields = $('[data-auto-save]');
        if (autoSaveFields.length === 0) return;

        var saveData = this.debounce(function() {
            var data = {};
            autoSaveFields.each(function() {
                var field = $(this);
                data[field.attr('name')] = field.val();
            });
            
            localStorage.setItem('heartapp_autosave', JSON.stringify(data));
            HeartApp.showNotification('Draft saved', 'info');
        }, this.config.debounceDelay);

        autoSaveFields.on('input change', saveData);

        // Restore saved data
        this.restoreAutoSaveData();
    },

    // Restore auto-saved data
    restoreAutoSaveData: function() {
        var savedData = localStorage.getItem('heartapp_autosave');
        if (savedData) {
            try {
                var data = JSON.parse(savedData);
                $.each(data, function(name, value) {
                    var field = $('[name="' + name + '"]');
                    if (field.length > 0 && field.val() === '') {
                        field.val(value);
                    }
                });
            } catch (e) {
                console.warn('Could not restore auto-saved data:', e);
            }
        }
    },

    // Utility: Debounce function
    debounce: function(func, wait) {
        var timeout;
        return function executedFunction() {
            var later = function() {
                clearTimeout(timeout);
                func.apply(this, arguments);
            }.bind(this);
            
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Utility: Format numbers
    formatNumber: function(num, decimals = 1) {
        return Number(num).toFixed(decimals);
    },

    // Utility: Format date
    formatDate: function(date, format = 'MM/DD/YYYY') {
        if (!(date instanceof Date)) {
            date = new Date(date);
        }
        
        var options = {};
        switch (format) {
            case 'MM/DD/YYYY':
                options = { month: '2-digit', day: '2-digit', year: 'numeric' };
                break;
            case 'YYYY-MM-DD':
                return date.toISOString().split('T')[0];
            default:
                options = { year: 'numeric', month: 'long', day: 'numeric' };
        }
        
        return date.toLocaleDateString('en-US', options);
    },

    // Utility: Smooth scroll to element
    scrollTo: function(target, offset = 0) {
        var element = $(target);
        if (element.length > 0) {
            $('html, body').animate({
                scrollTop: element.offset().top - offset
            }, this.config.animationDuration);
        }
    },

    // Utility: Copy text to clipboard
    copyToClipboard: function(text) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(function() {
                HeartApp.showNotification('Copied to clipboard', 'success');
            }).catch(function(err) {
                console.error('Could not copy text: ', err);
                HeartApp.showNotification('Failed to copy text', 'danger');
            });
        } else {
            // Fallback for older browsers
            var textArea = document.createElement("textarea");
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand('copy');
                HeartApp.showNotification('Copied to clipboard', 'success');
            } catch (err) {
                console.error('Fallback: Could not copy text: ', err);
                HeartApp.showNotification('Failed to copy text', 'danger');
            }
            document.body.removeChild(textArea);
        }
    },

    // Health-specific utilities
    health: {
        // Risk level color mapping
        getRiskColor: function(riskLevel) {
            var colors = {
                'Low': 'success',
                'Medium': 'warning',
                'High': 'danger'
            };
            return colors[riskLevel] || 'secondary';
        },

        // Risk level icon mapping
        getRiskIcon: function(riskLevel) {
            var icons = {
                'Low': 'fa-check-circle',
                'Medium': 'fa-exclamation-triangle',
                'High': 'fa-exclamation-triangle'
            };
            return icons[riskLevel] || 'fa-question-circle';
        },

        // Format confidence score
        formatConfidence: function(score) {
            return HeartApp.formatNumber(score * 100) + '%';
        },

        // Validate health data ranges
        validateHealthValue: function(type, value) {
            var ranges = {
                'age': { min: 1, max: 120 },
                'resting_bp': { min: 50, max: 300 },
                'cholesterol': { min: 50, max: 1000 },
                'max_hr': { min: 60, max: 250 },
                'oldpeak': { min: 0, max: 10 }
            };
            
            var range = ranges[type];
            if (range) {
                return value >= range.min && value <= range.max;
            }
            return true;
        }
    }
};

// Export to global scope
window.HeartApp = HeartApp;

// Additional event listeners for specific pages
$(document).ready(function() {
    
    // Dashboard specific functionality
    if ($('.dashboard-card').length > 0) {
        $('.dashboard-card').on('click', function() {
            var href = $(this).data('href');
            if (href) {
                window.location.href = href;
            }
        });
    }

    // Prediction form enhancements
    if ($('#predictionForm').length > 0) {
        // Add info popovers for medical terms
        $('[data-medical-term]').each(function() {
            var term = $(this).data('medical-term');
            var definitions = {
                'cholesterol': 'A waxy substance found in blood. High levels can increase heart disease risk.',
                'angina': 'Chest pain caused by reduced blood flow to the heart muscle.',
                'ecg': 'Electrocardiogram - a test that measures electrical activity of the heart.',
                'oldpeak': 'ST depression induced by exercise relative to rest.'
            };
            
            if (definitions[term]) {
                $(this).attr('data-bs-toggle', 'popover')
                       .attr('data-bs-content', definitions[term])
                       .attr('data-bs-trigger', 'hover');
            }
        });
    }

    // Chat interface enhancements
    if ($('#chatMessages').length > 0) {
        // Auto-scroll chat to bottom
        function scrollChatToBottom() {
            var chatContainer = $('#chatMessages');
            chatContainer.scrollTop(chatContainer[0].scrollHeight);
        }
        
        // Initial scroll
        scrollChatToBottom();
        
        // Scroll on new messages
        var observer = new MutationObserver(scrollChatToBottom);
        observer.observe(document.getElementById('chatMessages'), {
            childList: true,
            subtree: true
        });
    }

    // Print functionality
    $('.print-btn').on('click', function() {
        window.print();
    });

    // Export functionality
    $('.export-btn').on('click', function() {
        var format = $(this).data('format') || 'pdf';
        HeartApp.showNotification('Export functionality would be implemented here', 'info');
    });
});

// Service Worker registration for PWA functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}
