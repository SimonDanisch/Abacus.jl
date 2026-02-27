#!/usr/bin/env julia
#
# Stage 3: Ray Tracing Integration Tests
#
# Run with: julia --project=/sim/Programmieren/RayTracing test/raytracing_test.jl
# NOTE: Requires the main RayTracing project environment (not Abacus).
#
# Tests end-to-end ray tracing on the Vulkan backend:
# - Scene construction with meshes, materials, lights
# - BVH acceleration structure
# - Rendering via path tracing integrator
# - Pixel-level correctness checks

using Test
using Hikari, Raycore, GeometryBasics, StaticArrays, LinearAlgebra, Colors
using GeometryBasics: Rect3f, Sphere
using Hikari: RGBSpectrum
using Abacus

@testset "Stage 3: Ray Tracing Integration (Vulkan)" begin

    # ─────────────────────────────────────────────────────────────────────
    # Simple scene: ground plane + diffuse sphere + point light
    # Tests: BVH build, material evaluation, direct lighting
    # ─────────────────────────────────────────────────────────────────────
    @testset "3.1 — Simple diffuse sphere render" begin
        function make_mesh(prim; tess=32)
            prim = prim isa Sphere ? Tesselation(prim, tess) : prim
            mesh = normal_mesh(prim)
            return Hikari.TriangleMesh(mesh)
        end

        # Ground plane
        ground_geo = Rect3f(Point3f(-5, -5, -0.01), Vec3f(10, 10, 0.02))
        ground_mat = Hikari.MatteMaterial(
            Hikari.ConstantTexture(RGBSpectrum(0.5f0)),
            Hikari.ConstantTexture(0f0)
        )

        # Sphere
        sphere_geo = Sphere(Point3f(0, 0, 0.5), 0.5f0)
        sphere_mat = Hikari.MatteMaterial(
            Hikari.ConstantTexture(RGBSpectrum(0.9f0, 0.3f0, 0.15f0)),
            Hikari.ConstantTexture(0f0)
        )

        # Light
        light = Hikari.PointLight(Point3f(2, 2, 3), RGBSpectrum(20f0))

        # Camera
        eye = Point3f(3, 3, 2)
        look = Point3f(0, 0, 0.3)
        up = Vec3f(0, 0, 1)
        cam = Hikari.PerspectiveCamera(eye, look, up, 45f0, 64, 64)

        # Build scene
        prims = [
            Hikari.GeometricPrimitive(make_mesh(ground_geo), ground_mat),
            Hikari.GeometricPrimitive(make_mesh(sphere_geo), sphere_mat),
        ]
        scene = Hikari.Scene(prims, [light])

        # Render on Vulkan backend
        integrator = Hikari.DirectLightingIntegrator(cam, scene;
            backend=Abacus.VulkanBackend(), spp=1)
        img = Hikari.render(integrator)

        # Basic sanity checks
        @test size(img) == (64, 64)
        @test eltype(img) <: AbstractRGB
        # Image should not be all black (light is hitting the sphere)
        pixel_sum = sum(red.(img)) + sum(green.(img)) + sum(blue.(img))
        @test pixel_sum > 0
        # Image should not be all white/saturated
        @test pixel_sum < 64 * 64 * 3
    end

    # ─────────────────────────────────────────────────────────────────────
    # Mirror sphere — tests specular reflection BSDF
    # ─────────────────────────────────────────────────────────────────────
    @testset "3.2 — Mirror sphere" begin
        function make_mesh(prim; tess=32)
            prim = prim isa Sphere ? Tesselation(prim, tess) : prim
            mesh = normal_mesh(prim)
            return Hikari.TriangleMesh(mesh)
        end

        ground_mat = Hikari.MatteMaterial(
            Hikari.ConstantTexture(RGBSpectrum(0.5f0)),
            Hikari.ConstantTexture(0f0)
        )
        mirror_mat = Hikari.MirrorMaterial(
            Hikari.ConstantTexture(RGBSpectrum(0.95f0))
        )

        ground = Hikari.GeometricPrimitive(
            make_mesh(Rect3f(Point3f(-5,-5,-0.01), Vec3f(10,10,0.02))), ground_mat)
        sphere = Hikari.GeometricPrimitive(
            make_mesh(Sphere(Point3f(0,0,0.5), 0.5f0)), mirror_mat)

        light = Hikari.PointLight(Point3f(2, 2, 3), RGBSpectrum(20f0))
        cam = Hikari.PerspectiveCamera(
            Point3f(2,2,1.5), Point3f(0,0,0.3), Vec3f(0,0,1), 45f0, 32, 32)

        scene = Hikari.Scene([ground, sphere], [light])
        integrator = Hikari.DirectLightingIntegrator(cam, scene;
            backend=Abacus.VulkanBackend(), spp=1)
        img = Hikari.render(integrator)

        @test size(img) == (32, 32)
        pixel_sum = sum(red.(img)) + sum(green.(img)) + sum(blue.(img))
        @test pixel_sum > 0
    end

    # ─────────────────────────────────────────────────────────────────────
    # CPU vs Vulkan comparison — renders should match within tolerance
    # ─────────────────────────────────────────────────────────────────────
    @testset "3.3 — CPU vs Vulkan consistency" begin
        function make_mesh(prim; tess=32)
            prim = prim isa Sphere ? Tesselation(prim, tess) : prim
            mesh = normal_mesh(prim)
            return Hikari.TriangleMesh(mesh)
        end

        mat = Hikari.MatteMaterial(
            Hikari.ConstantTexture(RGBSpectrum(0.8f0)),
            Hikari.ConstantTexture(0f0)
        )
        ground = Hikari.GeometricPrimitive(
            make_mesh(Rect3f(Point3f(-3,-3,-0.01), Vec3f(6,6,0.02))), mat)
        sphere = Hikari.GeometricPrimitive(
            make_mesh(Sphere(Point3f(0,0,0.5), 0.5f0)), mat)

        light = Hikari.PointLight(Point3f(1, 1, 2), RGBSpectrum(10f0))
        cam = Hikari.PerspectiveCamera(
            Point3f(2,2,1.5), Point3f(0,0,0.3), Vec3f(0,0,1), 45f0, 16, 16)

        scene = Hikari.Scene([ground, sphere], [light])

        # Render on CPU
        cpu_integrator = Hikari.DirectLightingIntegrator(cam, scene; spp=1)
        cpu_img = Hikari.render(cpu_integrator)

        # Render on Vulkan
        vk_integrator = Hikari.DirectLightingIntegrator(cam, scene;
            backend=Abacus.VulkanBackend(), spp=1)
        vk_img = Hikari.render(vk_integrator)

        # Compare — should match within floating point tolerance
        # (same scene, same seed, same integrator logic)
        cpu_r = Float32.(red.(cpu_img))
        vk_r = Float32.(red.(vk_img))
        @test isapprox(cpu_r, vk_r; rtol=0.05)
    end
end
