#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("0 == 0", "[tests]")
{
	REQUIRE(0 == 0);
}

TEST_CASE("1 + 1 == 2", "[tests]")
{
	REQUIRE(1 + 1 == 2);
}