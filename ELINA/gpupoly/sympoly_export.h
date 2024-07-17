
#ifndef SYMPOLY_EXPORT_H
#define SYMPOLY_EXPORT_H

#ifdef SYMPOLY_STATIC_DEFINE
#  define SYMPOLY_EXPORT
#  define SYMPOLY_NO_EXPORT
#else
#  ifndef SYMPOLY_EXPORT
#    ifdef sympoly_EXPORTS
        /* We are building this library */
#      define SYMPOLY_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define SYMPOLY_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef SYMPOLY_NO_EXPORT
#    define SYMPOLY_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef SYMPOLY_DEPRECATED
#  define SYMPOLY_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef SYMPOLY_DEPRECATED_EXPORT
#  define SYMPOLY_DEPRECATED_EXPORT SYMPOLY_EXPORT SYMPOLY_DEPRECATED
#endif

#ifndef SYMPOLY_DEPRECATED_NO_EXPORT
#  define SYMPOLY_DEPRECATED_NO_EXPORT SYMPOLY_NO_EXPORT SYMPOLY_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef SYMPOLY_NO_DEPRECATED
#    define SYMPOLY_NO_DEPRECATED
#  endif
#endif

#endif /* SYMPOLY_EXPORT_H */
