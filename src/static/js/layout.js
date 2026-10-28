document.addEventListener('DOMContentLoaded', function () {
    const backToTopBtn = document.getElementById('backToTop');

    window.addEventListener('scroll', function () {
        if (window.scrollY > 300) {
        backToTopBtn.classList.add('show');
        } else {
        backToTopBtn.classList.remove('show');
        }
    });

    backToTopBtn.addEventListener('click', function () {
        window.scrollTo({
        top: 0,
        behavior: 'smooth'
        });
    });
});


document.addEventListener('DOMContentLoaded', function () {
    const mainFab = document.getElementById('mainFab');
    const fabOptions = document.getElementById('fabOptions');

    mainFab.addEventListener('click', function (e) {
        e.stopPropagation();
        fabOptions.classList.toggle('show');
        mainFab.classList.toggle('active');
    });

    document.addEventListener('click', function () {
        if (fabOptions.classList.contains('show')) {
        fabOptions.classList.remove('show');
        mainFab.classList.remove('active');
        }
    });
});
